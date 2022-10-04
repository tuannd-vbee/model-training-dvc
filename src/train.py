import numpy as np
from datasets import load_dataset, load_metric

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from dvclive.huggingface import DvcLiveCallback
from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from model import Wav2Vec2Continual
from transformers import Wav2Vec2CTCTokenizer

import torch
import torchaudio

vocab_file = 'data/vocab_vi.json'
new_data_inferenced_path = 'new_data_inferenced.csv'
eval_data_path = 'data/csv/test.csv'
pretrained_checkpoint = '/home3/tuannd/asr-training/artifacts/checkpoint-180000'

train_dataset = load_dataset("csv", data_files=new_data_inferenced_path, split="train", cache_dir='./.cache')
test_dataset = load_dataset("csv", data_files=eval_data_path, split="train", cache_dir='./.cache')

chars_to_ignore_regex = '[\,\̀\#\̃\_\̣\=\$\&\̉\?\̀\(\)\+\/\_\'\"\{\}\<\>\|\`\~\*\&\^\@\.\!\́\-\;\:\"\“\%\‘\”\�]'

tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["transcript"]
    batch["target_text_old"] = batch["transcript_old"]
    return batch

train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(speech_file_to_array_fn, remove_columns=test_dataset.column_names)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
        batch["old_labels"] = processor(batch["target_text_old"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batch_size=8, num_proc=8, batched=True)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, batch_size=8, num_proc=8, batched=True)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        old_label_features = [{"input_ids": feature["old_labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

            old_labels_batch = self.processor.pad(
                old_label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        old_labels = old_labels_batch["input_ids"].masked_fill(old_labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["old_labels"] = old_labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2Continual.from_pretrained(
    pretrained_checkpoint, 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
  output_dir="checkpoints",
  group_by_length=False,
  per_device_train_batch_size=2,
  evaluation_strategy="steps",
  num_train_epochs=1,
  save_steps=4,
  eval_steps=4,
  logging_steps=4,
  dataloader_num_workers=6,
  learning_rate=1e-5,
  warmup_steps=0,
  save_total_limit=10,
  eval_accumulation_steps=1,
  report_to='none',
  no_cuda=True
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.add_callback(DvcLiveCallback())
# trainer.train(resume_from_checkpoint=True)
trainer.train()

metrics = trainer.evaluate()

# import json
# with open("metrics.json", "w") as f:
#     json.dump(metrics, f, indent=4)