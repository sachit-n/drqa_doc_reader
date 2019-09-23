import json
import numpy as np
import torch
from model import StanfAR
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import time

'''
steps - 
1. load data
2. preprocess
3. train
4. tensorboard / evaluation on dev
5. saving/checkpointing/loading model
6. predict function
7. web app
8. packaging, code quality testing, etc.
'''
#%%


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


#%% loading
word2idx, idx2word, word_emb, train_data, dev_data = load_files(path='data')

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% preprocessing
train_q = torch.LongTensor(train_data['ques_idxs']).to(device)
train_c = torch.LongTensor(train_data['context_idxs']).to(device)

labels1 = torch.as_tensor(train_data['y1s']).to(device)
labels2 = torch.as_tensor(train_data['y2s']).to(device)

word_emb = torch.as_tensor(word_emb).to(device)

#%%
dev_q = torch.LongTensor(dev_data['ques_idxs']).to(device)
dev_c = torch.LongTensor(dev_data['context_idxs']).to(device)

labels1_dev = torch.as_tensor(dev_data['y1s']).to(device)
labels2_dev = torch.as_tensor(dev_data['y2s']).to(device)


#%%
class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = (train_q, train_c, labels1, labels2, dev_q, dev_c, labels1_dev, labels2_dev)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        query = self.data[0][idx]
        ctx = self.data[1][idx]
        y1 = self.data[2][idx]
        y2 = self.data[3][idx]

        dev_query = self.data[4][idx]
        dev_ctx = self.data[5][idx]
        dev_l1 = self.data[6][idx]
        dev_l2 = self.data[7][idx]

        return query, ctx, y1, y2, dev_query, dev_ctx, dev_l1, dev_l2


#%%
df = torch.utils.data.DataLoader(Dataset(), batch_size=32)


#%% training loop
torch.set_grad_enabled(True)

network = StanfAR(word_emb, 32).to(device)

optimizer = optim.Adamax(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

i = 0
num_epochs = 500

#%%
for j in range(num_epochs):
    test_acc1 = []
    test_acc2 = []
    acc1 = []
    acc2 = []
    i = 0
    tic_b = time.time()
    for batch in df:  # Get Batch
        i += 1
        query, context, y1, y2, dev_q, dev_ctx, dev_y1, dev_y2 = batch

        if query.shape[0] != 32:
            break

        if i == 100:
            toc_b = time.time()
            print(f"Time for 100 batches: {toc_b-tic_b}")

        preds = network(query, context)  # Pass Batch

        loss = (F.cross_entropy(preds[0], y1))+(F.cross_entropy(preds[1], y2))

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()

        acc1.append((preds[0].argmax(dim=1) == y1).sum())
        acc2.append((preds[0].argmax(dim=1) == y1).sum())

        with torch.no_grad():
            test_preds1, test_preds2 = network(dev_q, dev_ctx)
            accuracy1 = (test_preds1.argmax(dim=1)==dev_y1).sum()
            accuracy2 = (test_preds2.argmax(dim=1) == dev_y2).sum()
            test_acc1.append(accuracy1)
            test_acc2.append(accuracy2)

    print(f"Epoch: {j}\ntrain_accuracy1: {np.mean(acc1[-100:])}\ntrain_accuracy2: {np.mean(acc2[-100:])}\ntest_accuracy1: {np.mean(test_acc1[-100:])}\ntest_accuracy2: {np.mean(test_acc2[-100:])}\n")


#%%
print(
    "epoch:", 0,
    "total_correct:", total_correct,
    "loss:", total_loss
)



#%%
model = StanfAR(word_emb, 32)
sample_data_q, sample_data_c  = next(iter(train_loader_q)), next(iter(train_loader_c))

out = model(sample_data_q, sample_data_c)
