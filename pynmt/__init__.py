import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from google_drive_downloader import GoogleDriveDownloader

# デバイスの指定
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE :', DEVICE)

MODEL_DIR = "./content/model/"

class NMT(object):
    model: object
    def __init__(self, dir):
        self.trained_model = MT5ForConditionalGeneration.from_pretrained(dir).to(DEVICE)
        self.tokenizer = MT5Tokenizer.from_pretrained(dir, is_fast=True)
        additional_special_tokens = ['<A>', '<B>', '<C>', '<D>', '<E>', '<a>', '<b>', '<c>', '<d>', '<e>']
        self.tokenizer.add_tokens(additional_special_tokens)
        
    def translate(self, src_sentence: str):
        input_ids = self.tokenizer(src_sentence, return_tensors='pt').input_ids
        predict = self.trained_model.generate(input_ids)
        return [self.tokenizer.decode(i, skip_special_tokens=True) for i in predict]
        
def make_pynmt(model_id='1qZmBK0wHO3OZblH8nabuWrrPXU6JInDc', model_file='./model.zip'):
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=model_id, dest_path=model_file, unzip=True)
    nmt = NMT(MODEL_DIR)

    def pynmt(sentence):
        pred = nmt.translate(sentence)
        return pred
    return pynmt

