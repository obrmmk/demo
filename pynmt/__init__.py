import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from google_drive_downloader import GoogleDriveDownloader

# デバイスの指定
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE :', DEVICE)

class NMT(object):
    model: object
    def __init__(self, model_file):
        trained_model = MT5ForConditionalGeneration.from_pretrained(model_file)

        tokenizer = MT5Tokenizer.from_pretrained(model_file, is_fast=True)
        additional_special_tokens = ['<A>', '<B>', '<C>', '<D>', '<E>', '<a>', '<b>', '<c>', '<d>', '<e>']
        tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
        
        input_ids = tokenizer(text, return_tensors='pt').input_ids
        if USE_GPU:
            input_ids = input_ids.cuda()
        predict = trained_model.generate(input_ids)

    def translate_beam(self, src_sentence: str, beamsize=5):
        """
        複数の翻訳候補をリストで返す。
        """
        pred_list = self.predict

def make_pynmt(model_id='1qZmBK0wHO3OZblH8nabuWrrPXU6JInDc', model_file='./model.zip'):
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=model_id, dest_path=model_file, unzip=True)
    nmt = NMT("./content/model/")

    def pynmt(sentence):
        pred, prob = nmt.translate_beam(sentence)
        return pred, prob
    return pynmt