
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from model import Wav2Vec2Continual
import pandas as pd
from jiwer import wer
import kenlm
import librosa
from pyctcdecode import build_ctcdecoder
import re
import time

device = 'cuda'

def compute_wer(path_model, path_csv, delimiter):
    # load pretrained model
    tokenizer = Wav2Vec2Processor.from_pretrained(path_model)
    model = Wav2Vec2ForCTC.from_pretrained(path_model).to(device)
    model.eval()

    # test = pd.read_csv(path_csv)
    test = pd.read_csv(path_csv, delimiter=delimiter)

    transcript = []
    predict = []

    # specify alphabet labels as they appear in logits
    labels = ["ắ", "ồ", "z", "ứ", "ỡ", "ì", "x", "ặ", "u", "ẹ", "d", "ỵ", "r", "p", "t", "ỳ", "ẩ", "f", "ó", "á", "v", "ã", "i", "ư", "ở", "ễ", "ụ", "ú", "ũ", " ", "ă", "é", "ằ", "a", "ấ", "ờ", "ữ", "ớ", "n", "ý", "s", "h", "ơ", "ị", "l", "c", "k", "ỷ", "ỗ", "ế", "ẻ", "ợ", "ẫ", "í", "ỏ", "ủ", "g", "q", "j", "ò", "ỹ", "ự", "ô", "b", "y", "ĩ", "ỉ", "ẵ", "ầ", "ê", "ộ", "ậ", "m", "ń", "o", "ọ", "đ", "ẽ", "ử", "à", "è", "e", "ẳ", "ổ", "ù", "w", "ả", "ạ", "â", "ệ", "ề", "õ", "ố", "ể", "ừ", "-"]

    kenlm_model = kenlm.Model('/home3/thanhpv/contextualized_asr/gram3_addr.bin')
    # prepare decoder and decode logits via shallow fusion

    alpha = [0.5]
    beta = [5]

    for a in alpha:
        for b in beta:
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path='/home3/thanhpv/contextualized_asr/gram3_addr.bin',
                alpha=a,
                beta=b,
            )

            hotwords = open('/home3/cuongld/ASR_team/data_ASR/backup_scale_asr_server/scale_asr/asr-resources/transcript/hotword/hotwords_general.txt', 'r').read().split('\n')
            lm_decode_time = []

            for i in range(test.shape[0]):
                    try:
                        if librosa.core.get_duration(filename=test['path'][i]) < 15.0:
                            if str(test['transcript'][i]) in ['', ' ', 'nan']:
                                test['transcript'][i] = 'im lặng'
                            audio_input, _ = librosa.load(test['path'][i], sr=16000)
                            # transcribe
                            input_values = tokenizer(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
                            logits = model(input_values).logits
                            logits = logits.cpu().detach().numpy()[0]

                            a = time.perf_counter()

                            prediction = decoder.decode(logits,\
                                # beam_width=100, token_min_logp=-10, beam_prune_logp=-10, hotwords=hotwords, hotword_weight=5.0,)\
                            beam_width=10, token_min_logp=-10, beam_prune_logp=-10)\
                                .replace('-', ' ').replace('  ', ' ').strip()

                            a = time.perf_counter() - a
                            lm_decode_time.append(a)

                            prediction = (' ' + prediction.lower() + ' ')
                            # pred_words = []
                            # for word in prediction.split(' '):
                            #     if len(word) == 2:
                            #         if word[-1] in ['y', 'ỹ', 'ỷ', 'ý', 'ỳ', 'ỵ'] and word[0] not in ['ấ', 'â', 'ầ', 'ẫ', 'ậ', 'ẩ', 'a', 'á', 'à', 'ạ', 'ả']:
                            #             word = word.replace('y', 'i').replace('ý', 'í').replace('ỹ', 'ĩ')\
                            #                                 .replace('ỳ', 'ì').replace('ỷ', 'ỉ')
                            #     if word == 'à' or word == 'ờ':
                            #         word = ''
                            #     pred_words.append(word)
                            
                            # prediction = ' '.join(pred_words)
                            prediction = prediction.replace('w', '').replace('f', '')
                            prediction = re.sub(' +', ' ', prediction).strip()

                            transcription = (' ' + test['transcript'][i].lower() + ' ')
                            # trans_words = []
                            # for word in transcription.split(' '):
                            #     if len(word) == 2:
                            #         if word[-1] in ['y', 'ỹ', 'ỷ', 'ý', 'ỳ', 'ỵ'] and word[0] not in ['ấ', 'â', 'ầ', 'ẫ', 'ậ', 'ẩ', 'a', 'á', 'à', 'ạ', 'ả']:
                            #             word = word.replace('y', 'i').replace('ý', 'í').replace('ỹ', 'ĩ')\
                            #                                 .replace('ỳ', 'ì').replace('ỷ', 'ỉ')
                            #     # if word == 'à' or word == 'ờ':
                            #     #     word = ''
                            #     trans_words.append(word)
                            
                            # transcription = ' '.join(trans_words)
                            transcription = re.sub(' +', ' ', transcription).strip()
                            if prediction in ['', ' ']:
                                prediction = 'im lặng'

                            print('------')
                            print(i)
                            print(prediction)
                            print(transcription)

                            transcript.append(transcription)
                            predict.append(prediction)
                            # print(i)
                    except:
                        bruh=0
            if len(predict) < 2500:
                error = wer(transcript, predict)
                print('alpha', a)
                print('beta', b)
                print(error)
            else:
                error = 0
                range_test = int(len(predict) - (len(predict) % 1))

                for i in range(0, range_test, 1):
                    error += wer(transcript[i:i+1], predict[i:i+1])
                
                error /= (range_test / 1)
                print('alpha', a)
                print('beta', b)
                print(error)
            print(sum(lm_decode_time)/len(lm_decode_time))
    

# compute_wer('/home3/thanhpv/checkpoint-buoc1/checkpoint-last', '/home3/thanhpv/data_csv/processed/test_vimix_noisy.csv', delimiter='\t')

compute_wer('/home3/thanhpv/continual_learning/cont_addr_aicc/checkpoint-110000', '/home3/thanhpv/test_6h.csv', delimiter='\t')
# compute_wer('/home3/thanhpv/checkpoint-buoc1/checkpoint-last', '/home3/thanhpv/data_csv/processed/qa_test_norm.csv', delimiter='\t')
# compute_wer('/home3/thanhpv/checkpoint-buoc1/checkpoint-last', '/home3/thanhpv/data_csv/processed/thuam_test_norm.csv', delimiter='\t')
# compute_wer('/home3/thanhpv/checkpoint-buoc1/checkpoint-last', '/home3/thanhpv/data_csv/processed/test_vimix_clean.csv', delimiter='\t')
# compute_wer('/home3/thanhpv/checkpoint_1_all_augmentation/checkpoint-36000', '/home3/thanhpv/data_csv/processed/thuam_test_norm.csv', delimiter='\t')








