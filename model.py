import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from custom_layers import AlignedQuesEmb, QuesAttn

print("new2")

# todo - Additional passage features, concatenating lstm outputs, partially freeze pretrained


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_uncased_match(query, context):
    ctx_df = pd.DataFrame(context.tolist()).T
    out_tensor = torch.tensor([], dtype=torch.float32)
    query_checker = query.tolist()
    for i in range(query.shape[0]):
        em = torch.as_tensor((ctx_df[i].isin(query_checker[i])) & (ctx_df[i].isin([0, 1])==False), dtype=torch.float32).unsqueeze(0)
        out_tensor = torch.cat([out_tensor, em], dim=0)
    return out_tensor.to(device)


# #%%
# def get_ctx_features(query, context):
#     return query, context
#
#
# #%%
# samp = next(iter(df))
#
# q = samp[0]
# c = samp[1]


#%%
class StanfAR(nn.Module):
    def __init__(self, word_emb):
        super().__init__()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings=word_emb)
        self.dropout = nn.Dropout(p=0.3)

        self.query_align = AlignedQuesEmb().to(device)

        self.query_attn = QuesAttn().to(device)

        self.lstm_query = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
        self.lstm_ctx = nn.LSTM(input_size=601, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)

        self.w_start = torch.nn.Parameter(torch.randn(256, 256).to(device))
        self.w_end = torch.nn.Parameter(torch.randn(256, 256).to(device))

    def forward(self, X, Y):

        query = X
        ctx = Y

        exact_match = get_uncased_match(query, ctx).reshape(query.shape[0], 400, 1)

        # print(f"Input Shape - {query.shape}")

        query_vectorized = self.embedding_layer(query)
        # print(f"Embedding Layer Shape - {query_vectorized.shape}")
        query_emb = self.dropout(query_vectorized)

        ctx_vectorized = self.embedding_layer(ctx)
        ctx_emb = self.dropout(ctx_vectorized)

        aligned_ques_emb = self.query_align(query_emb, ctx_emb)

        ctx_features = torch.cat([ctx_emb, aligned_ques_emb], dim=2)
        ctx_features = torch.cat([ctx_features, exact_match], dim=2)

        query_lstm_out = self.lstm_query(query_emb)[0]
        # print(f"Query LSTM Out shape - {query_lstm_out.shape}")

        attn_query = self.query_attn(query_lstm_out)
        # print(f"{attn_query.shape}")

        ctx_lstm_out = self.lstm_ctx(ctx_features)[0]
        # print(f"ctxLSTM shape - {ctx_lstm_out.shape}")

        qtW_start = torch.matmul(attn_query, self.w_start)
        qtW_end = torch.matmul(attn_query, self.w_end)

        start_token = torch.bmm(qtW_start, ctx_lstm_out.permute(0, 2, 1))
        end_token = torch.bmm(qtW_end, ctx_lstm_out.permute(0, 2, 1))

        return start_token.squeeze(), end_token.squeeze()


#%%



#%%
#Debugging


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dropout = nn.Dropout(p=0.5)
#         self.embedding_layer = nn.Embedding.from_pretrained(embeddings=word_emb)
#         self.align_layer = nn.Linear(300, 150)
#
#     def forward(self, x):
#         x = x
#         query_vectorized = self.embedding_layer(x)
#         out = self.dropout(query_vectorized)
#         out = self.align_layer(out)
#
#         return out
# #
# #
# #%%
# t = Net()
#
# s = torch.as_tensor([[1,2,3,4,5]])
#
# s
# #%%
# o = t(s)


# #%%
# train_q = torch.LongTensor(train_data['ques_idxs'])
# train_c = torch.LongTensor(train_data['context_idxs'])
#
#
# #%%
#
# qt = train_q[:10, :]
# ct = train_c[:10, :]
#
# #%%
# m = StanfAR(word_emb)
#
# o = m(qt, ct)
#
# #%%
#
#
# class TempNN(nn.Module):
#     def __init__(self, word_emb):
#         super().__init__()
#         self.embedding_layer = nn.Embedding.from_pretrained(embeddings=word_emb)
#         self.lstm_query = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
#
#     def forward(self, X):
#         query_vector_out = self.embedding_layer(X)
#         query_lstm_out, _ = self.lstm_query(query_vector_out)
#         print(query_lstm_out.shape)
#
#         return query_lstm_out
#
#
# #%%
# word_emb = torch.FloatTensor(word_emb)
#
# train = torch.LongTensor(train_data['ques_idxs'])
#
#
# #%%
# network = TempNN(word_emb)
#
# tr = train[:10, :]
#
# emb = network(tr)
#
#
# #%%
# t = torch.tensor([[1, 1, 1], [10, 0.2, 5]])
#
# #%%
# n = TempNN()
#
# n(t.float())
#
#
# #%%
#
# pret = torch.load("data/glove.840B.300d/glove.840B.300d.txt")