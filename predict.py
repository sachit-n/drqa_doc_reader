import spacy
import torch
from model import StanfAR
import json
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json_file(path):
    with open(path) as file:
        out = json.load(file)
    return out


def load_npz_file(path):
    return np.load(path)


def load_files(path):
    word2idx = load_json_file(path + "/word2idx.json")
    word_emb = load_json_file(path + "/word_emb.json")
    word_emb = torch.as_tensor(word_emb).to(device)

    idx2word = {i:j for j,i in word2idx.items()}

    return word2idx, idx2word, word_emb


#%% loading
word2idx, idx2word, word_emb = load_files(path='data')

spacy_parser = spacy.load("en")
model = StanfAR(word_emb).to(device)
model.load_state_dict(torch.load("models/epoch1.pth", map_location=device))
model.eval()


#%%
def string_to_tensor(tokens, seq_length):
    out_tensor = torch.zeros(1, seq_length, dtype=torch.int64).to(device)

    idxs = []
    for word in tokens:
        try:
            idxs.append(word2idx[word])
        except KeyError:
            idxs.append(1)

    out_tensor[:, :len(idxs)] = torch.as_tensor(idxs)
    return out_tensor


def find_next_best(pred: tuple, start_idx: int, end_idx: int) -> tuple:
    start_idx_new = pred[0].argsort[1].item()
    end_idx_new = pred[1].argsort[1].item()




def get_pred_idx(pred: tuple) -> tuple:
    start_idx = pred[0].argmax().item()
    end_idx = pred[1].argmax().item()

    if end_idx - start_idx > 15 or end_idx - start_idx < 0:
        find_next_best(pred, start_idx, end_idx)
        get_pred_idx(pred)

    return start_idx, end_idx


def pred_to_ans(pred: tuple, ctx_seq: torch.tensor) -> str:
    start_idx, end_idx = get_pred_idx(pred)

    ans = ''
    for idx in range(start_idx, end_idx+1):
        word_idx = ctx_seq[idx].item()
        ans = ' ' + idx2word[word_idx]

    return ans


def predict(query: str, context: str) -> str:
    """
    predict answer for a single question, para pair
    :param query: query string
    :param context: paragraph string
    :return: answer string
    """
    global ctx_seq, prediction
    query_tokens = [token.text for token in spacy_parser(query)]
    ctx_tokens = [token.text for token in spacy_parser(context)]

    query_seq = string_to_tensor(query_tokens, 50)
    ctx_seq = string_to_tensor(ctx_tokens, 400)

    with torch.no_grad():
        prediction = F.softmax(model(query_seq, ctx_seq))

    predicted_ans = pred_to_ans(prediction, ctx_seq.squeeze(0))

    return predicted_ans








