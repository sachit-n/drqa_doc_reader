import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuesAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentinel_vec = torch.nn.Parameter(torch.randn(256,1)).to(device)

    def forward(self, query_lstm_out):
        attn = torch.matmul(query_lstm_out, self.sentinel_vec)
        attn_wts = F.softmax(attn, dim=1)

        query_weighted = torch.bmm(attn_wts.permute(0, 2, 1), query_lstm_out)

        return query_weighted


class AlignedQuesEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.align_layer = nn.Linear(300, 300)

    def forward(self, query_emb, ctx_embed):
        query_emb_dense = F.relu(self.align_layer(query_emb))
        ctx_emb_dense = F.relu(self.align_layer(ctx_embed))

        emb_dot_prod = torch.bmm(ctx_emb_dense, query_emb_dense.permute(0, 2, 1))

        align_wts = F.softmax(emb_dot_prod, dim=2)

        aligned_emb = torch.bmm(align_wts, query_emb)

        return aligned_emb




#
# #%%Debugging
#
#
# class StanfAR(nn.Module):
#     def __init__(self, word_emb, batch_size):
#         super().__init__()
#
#         self.batch_size = batch_size
#
#         self.embedding_layer = nn.Embedding.from_pretrained(embeddings=word_emb)
#         self.dropout = nn.Dropout(p=0.3)
#
#         self.query_attention_sentinel = torch.nn.Parameter(torch.randn(1, 256))
#
#         self.lstm_query = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
#         self.lstm_ctx = nn.LSTM(input_size=300, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
#
#         self.w_start = torch.nn.Parameter(torch.randn(256, 256))
#         self.w_end = torch.nn.Parameter(torch.randn(256, 256))
#
#     def forward(self, X, Y):
#
#         query = X
#         ctx = Y
#
#         # print(f"Input Shape - {query.shape}")
#
#         query_vectorized = self.embedding_layer(query)
#         # print(f"Embedding Layer Shape - {query_vectorized.shape}")
#         query_emb = self.dropout(query_vectorized)
#
#         ctx_vectorized = self.embedding_layer(ctx)
#         ctx_emb = self.dropout(ctx_vectorized)
#
#         return query_emb, ctx_emb
#
# #%%
#
# a = StanfAR(word_emb, 32)
#
# query_emb, ctx_emb = a(query, context)
#
# #%%
# b = AlignedQuesEmb()
#
# o = b(query_emb, ctx_emb)