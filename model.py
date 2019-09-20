import torch
import torch.nn as nn
import torch.nn.functional as F


class StanfAR(nn.Module):
    def __init__(self, word_emb):
        super().__init__()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings=word_emb)

        self.query_attention_sentinel = torch.nn.Parameter(torch.randn(1, 256))

        self.lstm_query = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
        self.lstm_ctx = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)

        self.w_start = torch.nn.Parameter(torch.randn(256, 256))
        self.w_end = torch.nn.Parameter(torch.randn(256, 256))

    def forward(self, X, Y):
        global t

        query = X
        ctx = Y

        print(f"Input Shape - {query.shape}")

        query_vectorized = self.embedding_layer(query)
        print(f"Embedding Layer Shape - {query_vectorized.shape}")

        ctx_vectorized = self.embedding_layer(ctx)

        query_lstm_out = self.lstm_query(query_vectorized)[0]
        print(f"Query LSTM Out shape - {query_lstm_out.shape}")

        t = self.query_attention_sentinel
        print(f"{self.query_attention_sentinel.shape}")

        query_attn_w = torch.cat(10*[self.query_attention_sentinel]).reshape(10, 256, 1)
        print(f"{query_attn_w.shape}")
        attn_w = torch.bmm(query_lstm_out, query_attn_w)

        attn_wts = F.softmax(attn_w, dim=2)
        print(f"attn_layer shape: {attn_wts.shape}")

        attn_query = torch.bmm(query_lstm_out.permute(0, 2, 1), attn_wts)
        print(f"{attn_query.shape}")

        ctx_lstm_out = self.lstm_ctx(ctx_vectorized)[0]
        print(f"ctxLSTM shape - {ctx_lstm_out.shape}")

        w_start_reshaped = torch.cat(10*[self.w_start]).reshape(10, 256, 256)
        w_end_reshaped = torch.cat(10 * [self.w_end]).reshape(10, 256, 256)

        qtW_start = torch.bmm(attn_query.permute(0,2,1), w_start_reshaped)
        qtW_end = torch.bmm(attn_query.permute(0,2,1), w_end_reshaped)
        print(f"{qtW_start.shape}")
        print(f"{qtW_end.shape}")

        start_token = F.softmax(torch.bmm(qtW_start, ctx_lstm_out.permute(0, 2, 1)), dim=2)
        end_token = F.softmax(torch.bmm(qtW_end, ctx_lstm_out.permute(0, 2, 1)), dim=2)

        return start_token, end_token



#Debugging

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