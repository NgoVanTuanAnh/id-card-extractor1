import torch
import torch.nn as nn
import torch.nn.functional as F
from src.crnn.config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def nested_parser(data):
    # Base case: if data is not a dictionary, return it as is
    if not isinstance(data, dict):
        return data
    # Recursive case: if data is a dictionary, create a class object with attributes
    else:
        class Config:
            pass
        for key, value in data.items():
            setattr(Config, key, nested_parser(value))
        return Config
    
def encode_text_batch(text_batch):

    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [CHAR2IDX[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens


def decode_predictions(text_batch_logits, idx2char):

    text_batch_tokens = F.softmax(
        text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T  # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx,
                               letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
