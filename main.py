import json
import numpy as np
import torch
#from model import StanfAR


def load_json_file(path):
    with open(path) as file:
        out = json.load(file)
    return out


def load_npz_file(path):
    return np.load(path)


def load_files(path):
    word2idx = load_json_file(path + "/word2idx.json")
    word_emb = load_json_file(path + "/word_emb.json")

    train_data = load_npz_file(path + "/train.npz")
    dev_data = load_npz_file(path + "/dev.npz")

    idx2word = {i:j for j,i in word2idx.items()}

    return word2idx, idx2word, word_emb, train_data, dev_data


word2idx, idx2word, word_emb, train_data, dev_data = load_files('data')

word_emb = torch.tensor(word_emb)



model = StanfAR(word_emb)


sample_data = torch.tensor([[0, 2]])

out = model(sample_data)
