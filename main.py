import json
import numpy as np
import torch
from model import StanfAR
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

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

#%% preprocessing
train_q = torch.LongTensor(train_data['ques_idxs'])
train_c = torch.LongTensor(train_data['context_idxs'])

labels1 = torch.as_tensor(train_data['y1s'])
labels2 = torch.as_tensor(train_data['y2s'])

word_emb = torch.as_tensor(word_emb)


#%%
class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = (train_q, train_c, labels1, labels2)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        query = self.data[0][idx]
        ctx = self.data[1][idx]
        y1 = self.data[2][idx]
        y2 = self.data[3][idx]

        return query, ctx, y1, y2


#%%
df = torch.utils.data.DataLoader(Dataset(), batch_size=32)


#%% training loop
torch.set_grad_enabled(True)

network = StanfAR(word_emb, 32)

optimizer = optim.Adamax(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

i = 0
num_epochs = 500

#%%
for j in range(1, num_epochs):
    for batch in df:  # Get Batch
        i+=1
        query, context, y1, y2 = batch
        #print(f"Query input shape: {query.shape}\nContext Input shape: {context.shape}\ny1 shape: {y1.shape}\ny2 shape: {y2.shape}")
        if query.shape[0] != 32:
            break

        preds = network(query, context)  # Pass Batch

        loss = (F.cross_entropy(preds[0], y1))+(F.cross_entropy(preds[1], y2))
        #print(f"loss: {loss.shape}")

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        print(f"\n\n\n\naverage loss: {total_loss/(i)}\naccuracy: {(preds[0].argmax(dim=1)==y1).sum()}\n\n\n")


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
