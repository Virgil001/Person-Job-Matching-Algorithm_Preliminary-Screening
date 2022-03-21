import os
import numpy as np
import jieba as jb
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import warnings
import dataset as ds
from torch import optim
from logger import Progbar

warnings.filterwarnings('ignore')

w2v_model = KeyedVectors.load_word2vec_format('./zh.vec')

EPOCH = config.epoch
MODEL_SAVE_PATH = config.LANGUAGEMODEL_SAVE_PATH

MAX_LEN = ds.get_max_seq_len()
BATCH_SIZE = config.batch_size


class Word_Level_Embedding(nn.Module):
    def __init__(self, embedding_size, max_seq_len):
        super(Word_Level_Embedding, self).__init__()
        self.emb_dim = embedding_size
        self.max_len = max_seq_len

    def matrix_padding(self, emb_matrix):
        seq_len = emb_matrix.shape[0]
        # if seq_len < 512:
        return np.vstack((emb_matrix, np.zeros((self.max_len - seq_len, 300))))
        # else:
        #     return emb_matrix[:511, :]

    def word2vec(self, text):
        vocabs = []
        embedding = np.array([])
        wrd_list = list(jb.cut(str(text)))

        for word in wrd_list:
            if word in w2v_model.key_to_index:
                vocabs.append(word)
                embedding = np.append(embedding, w2v_model[word])
        embedding_matrix = embedding.reshape(len(vocabs),
                                             self.emb_dim)  # shape: (seq_length，300)
        embedding_matrix = self.matrix_padding(embedding_matrix)
        embedding_matrix = torch.tensor(embedding_matrix)
        embedding_tensor = embedding_matrix.unsqueeze(0)  # 在第1维进行升维 batch维度
        return embedding_tensor

    def get_batch_tensor(self, jobduty, jobreq, workexp, projexp, batch=BATCH_SIZE):
        for k in range(batch):
            jobduty_embedding_tensor = self.word2vec(jobduty[k])
            jobreq_embedding_tensor = self.word2vec(jobreq[k])
            workexp_embedding_tensor = self.word2vec(workexp[k])
            projexp_embedding_tensor = self.word2vec(projexp[k])
            yield jobduty_embedding_tensor, jobreq_embedding_tensor, workexp_embedding_tensor, \
                  projexp_embedding_tensor

    def forward(self, jobduty, jobreq, wrokexp, projexp, batch=BATCH_SIZE):
        tensor_iter = self.get_batch_tensor(jobduty, jobreq, wrokexp, projexp)
        jobduty_tensor \
            , jobreq_tensor \
            , wrokexp_tensor \
            , projexp_tensor = torch.zeros((batch, self.max_len, self.emb_dim)) \
            , torch.zeros((batch, self.max_len, self.emb_dim)) \
            , torch.zeros((batch, self.max_len, self.emb_dim)) \
            , torch.zeros((batch, self.max_len, self.emb_dim))
        continue_iterator = True
        while continue_iterator:
            for i in range(batch):
                try:
                    jobduty_tensor[i, :, :] = next(tensor_iter)[0][i]
                    jobreq_tensor[i, :, :] = next(tensor_iter)[1][i]
                    wrokexp_tensor[i, :, :] = next(tensor_iter)[2][i]
                    projexp_tensor[i, :, :] = next(tensor_iter)[3][i]
                except StopIteration:
                    continue_iterator = False

        return jobduty_tensor, jobreq_tensor, wrokexp_tensor, projexp_tensor


class BiLSTMAttention(nn.Module):
    def __init__(self, config):
        super(BiLSTMAttention, self).__init__()
        self.embedding = Word_Level_Embedding(config.embedding_size, MAX_LEN)
        self.lstm = nn.LSTM(config.embedding_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()

        self.w1 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w2 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w3 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w4 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))

        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)
        # nn.init.uniform_(self.w1, -0.1, 0.1)
        # nn.init.uniform_(self.w2, -0.1, 0.1)
        # nn.init.uniform_(self.w3, -0.1, 0.1)
        # nn.init.uniform_(self.w4, -0.1, 0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text1, text2, text3, text4):
        emb1, emb2, emb3, emb4 = self.embedding(text1, text2, text3, text4)  # (max_len, batch_size
        # embedding_size)

        h1, _ = self.lstm(emb1)  # (batch_size, seq_len, hidden_size * 2)
        h2, _ = self.lstm(emb2)
        h3, _ = self.lstm(emb3)
        h4, _ = self.lstm(emb4)

        m1 = self.tanh1(h1)  # (batch_size, seq_len, hidden_size * 2)
        m2 = self.tanh1(h2)
        m3 = self.tanh1(h3)
        m4 = self.tanh1(h4)

        # 张量广播操作
        alpha1 = F.softmax(torch.matmul(m1, self.w1), dim=1)  # (batch_size, seq_len, 1)
        alpha2 = F.softmax(torch.matmul(m2, self.w2), dim=1)
        alpha3 = F.softmax(torch.matmul(m3, self.w3), dim=1)
        alpha4 = F.softmax(torch.matmul(m4, self.w4), dim=1)

        # 张量元素相乘，会发生张量广播使得张量的维度满足条件
        out1 = h1 * alpha1  # (batch_size, seq_len, hidden_size * 2)
        out2 = h2 * alpha2
        out3 = h3 * alpha3
        out4 = h4 * alpha4

        # torch.sum操作默认情况下不保持维度
        out1 = torch.sum(out1, 1)  # (batch_size,hidden_size * 2)
        out2 = torch.sum(out2, 1)
        out3 = torch.sum(out3, 1)
        out4 = torch.sum(out4, 1)

        out1 = self.tanh2(out1)
        out2 = self.tanh2(out2)
        out3 = self.tanh2(out3)
        out4 = self.tanh2(out4)

        # 张量相加
        # out = torch.cat((out1, out2, out3, out4), 1)
        out = out1 + out2 + out3 + out4

        out = self.fc(out)  # (batch_size,num_classes)
        out = self.fc2(out)
        # out = self.softmax(out)
        return out


def train_LM(config):

    print("Start training model...")

    # train the model

    model = BiLSTMAttention(config)
    param_groups = model.parameters()

    optimizer = optim.ASGD(param_groups)  # adagrad(param_groups)    # (param_groups)  # SGD(param_groups, lr=0.01, weight_decay=0.001)  # AdamW(param_groups, eps=1e-6, weight_decay=0.001)

    total_num, _, train_label, jobduty, jobreq, workexp, projexp = ds.MyDataSet()
    steps = ds.get_steps_per_epoch(total_num, BATCH_SIZE)

    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.6)
    criterion = nn.CrossEntropyLoss()

    model_save_path = os.path.join(MODEL_SAVE_PATH, "Languagemodel.pkl")
    model.train()

    progbar = Progbar(target=steps)

    for epoch in range(EPOCH):

        print("\nEpoch {} is running...".format(epoch))

        dataset_iterator = ds.get_iterator_batch(jobduty, jobreq, workexp, projexp, train_label)

        for i, iteration in enumerate(dataset_iterator):

            model.zero_grad()
            jobduty_text, jobreq_text, workexp_text, projexp_text = iteration[0], iteration[1], \
                                                                    iteration[2], iteration[3]
            labels = torch.tensor(iteration[4], dtype=torch.long)
            optimizer.zero_grad()

            logits = model(jobduty_text, jobreq_text, workexp_text, projexp_text)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            StepLR.step()
            progbar.update(i + 1, None, None, [("train loss", loss.item())])

            if i == steps - 1:
                break

        print("\nEpoch {} is over!\n".format(epoch))

    torch.save(model, model_save_path)
    print("\nTraining is over!\n")


if __name__ == "__main__":
    train_LM(config)
