import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
import csv
from tqdm import tqdm
import librosa

device = 'cuda'
new_data_path = 'data/csv/new_data.csv'
all_data_path = 'data/csv/all_data.csv'
pretrained_checkpoint = '/home3/tuannd/asr-training/artifacts/checkpoint-180000'
new_data_inferenced_path = 'new_data_inferenced.csv'

# load pretrained model
tokenizer = Wav2Vec2Processor.from_pretrained(pretrained_checkpoint)
model = Wav2Vec2ForCTC.from_pretrained(pretrained_checkpoint).to(device).eval()

# load audio
data = pd.read_csv(new_data_path, delimiter=',')
# data = os.listdir('/home3/thanhpv/preprocess_vietnamese/fuccthiswav')

with open(new_data_inferenced_path, 'w') as infer_csv:
    writer = csv.writer(infer_csv, delimiter=',', quoting=csv.QUOTE_NONE)
    writer.writerow(['transcript', 'transcript_old','path'])
    for i in tqdm(range(len(data))):
        link = data.path[i]
        try:
            if librosa.core.get_duration(filename=link) < 50:
                audio_input, sr = librosa.load(link, sr=16000)
                # transcribe
                input_values = tokenizer(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)

                prediction = tokenizer.batch_decode(predicted_ids)[0]
                writer.writerow([data.transcript[i], prediction.lower().replace('[pad]', ''), link])
        except:
            bruh = 0
