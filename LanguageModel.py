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

EPOCH = ds.get_epoch()
MODEL_SAVE_PATH = config.LANGUAGEMODEL_SAVE_PATH

MAX_LEN = ds.get_max_seq_len()
BATCH_SIZE = config.batch_size


class Word_Level_Embedding(nn.Module):
    def __init__(self, max_seq_len, embedding_size):
        super(Word_Level_Embedding, self).__init__()
        self.emb_dim = embedding_size
        self.max_len = max_seq_len

    def forward(self, jobduty, jobreq, wrokexp, projexp, batch=BATCH_SIZE):
        def get_batch_tensor(jobduty, jobreq, workexp, projexp, batch=BATCH_SIZE):

            def word2vec(text):

                def matrix_padding(emb_matrix):
                    seq_len = emb_matrix.shape[0]
                    return np.vstack((emb_matrix, np.zeros((self.max_len - seq_len, 300))))

                vocabs = []
                embedding = np.array([])
                wrd_list = list(jb.cut(str(text)))

                for word in wrd_list:
                    if word in w2v_model.key_to_index:
                        vocabs.append(word)
                        embedding = np.append(embedding, w2v_model[word])
                embedding_matrix = embedding.reshape(len(vocabs),
                                                     self.emb_dim)  
                embedding_matrix = matrix_padding(embedding_matrix)
                embedding_matrix = torch.tensor(embedding_matrix)
                embedding_tensor = embedding_matrix.unsqueeze(1) 
                return embedding_tensor

            for k in range(batch):
                jobduty_embedding_tensor = word2vec(jobduty[k])
                jobreq_embedding_tensor = word2vec(jobreq[k])
                workexp_embedding_tensor = word2vec(workexp[k])
                projexp_embedding_tensor = word2vec(projexp[k])
                yield jobduty_embedding_tensor, jobreq_embedding_tensor, workexp_embedding_tensor, \
                      projexp_embedding_tensor

        tensor_iter = get_batch_tensor(jobduty, jobreq, wrokexp, projexp)
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
        self.embedding = Word_Level_Embedding(MAX_LEN, config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()

        self.w1 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w2 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w3 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        self.w4 = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))

        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        nn.init.uniform_(self.w1, -0.1, 0.1)
        nn.init.uniform_(self.w2, -0.1, 0.1)
        nn.init.uniform_(self.w3, -0.1, 0.1)
        nn.init.uniform_(self.w4, -0.1, 0.1)

    def forward(self, text1, text2, text3, text4):
        emb1, emb2, emb3, emb4 = self.embedding(text1, text2, text3, text4)  # (batch_size, seq_len,
        # embedding_size)

        h1, _ = self.lstm(emb1)  # (batch_size, seq_len, hidden_size * 2)
        h2, _ = self.lstm(emb2)
        h3, _ = self.lstm(emb3)
        h4, _ = self.lstm(emb4)

        m1 = self.tanh1(h1)  # (batch_size, seq_len, hidden_size * 2)
        m2 = self.tanh1(h2)
        m3 = self.tanh1(h3)
        m4 = self.tanh1(h4)

        alpha1 = F.softmax(torch.matmul(m1, self.w1), dim=1)  # (batch_size, seq_len, 1)
        alpha2 = F.softmax(torch.matmul(m2, self.w2), dim=1)
        alpha3 = F.softmax(torch.matmul(m3, self.w3), dim=1)
        alpha4 = F.softmax(torch.matmul(m4, self.w4), dim=1)

        out1 = h1 * alpha1  # (batch_size, seq_len, hidden_size * 2)
        out2 = h2 * alpha2
        out3 = h3 * alpha3
        out4 = h4 * alpha4
        
        out1 = torch.sum(out1, 1)  # (batch_size,hidden_size * 2)
        out2 = torch.sum(out2, 1)
        out3 = torch.sum(out3, 1)
        out4 = torch.sum(out4, 1)

        out1 = self.tanh2(out1)
        out2 = self.tanh2(out2)
        out3 = self.tanh2(out3)
        out4 = self.tanh2(out4)

        out = out1 + out2 + out3 + out4

        out = self.fc(out)  # (batch_size,num_classes)
        return out


def train_LM(config):
    def get_steps_per_epoch(line_count, batch_size):
        return line_count // batch_size if line_count % batch_size == 0 else line_count // batch_size + 1

    print("Start training model...")
    # train the model

    model = BiLSTMAttention(config)
    param_groups = model.parameters()
    optimizer = optim.AdamW(param_groups, eps=1e-6, weight_decay=0.001, amsgrad=True)

    criterion = nn.CrossEntropyLoss()
    model_save_path = os.path.join(MODEL_SAVE_PATH, "Languagemodel.pkl")
    model.train()

    for epoch in range(EPOCH):

        print("\nEpoch {} is running...\n".format(epoch))

        total_num, _, train_label, jobduty, jobreq, workexp, projexp = ds.MyDataSet()
        steps = get_steps_per_epoch(total_num, config.batch_size)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.6)
        dataset_iterator = ds.get_iterator_batch(jobduty, jobreq, workexp, projexp, train_label)
        progbar = Progbar(target=steps)

        for i, iteration in enumerate(dataset_iterator):

            model.zero_grad()
            jobduty_text, jobreq_text, workexp_text, projexp_text = iteration[0], iteration[1], \
                                                                    iteration[2], iteration[3]
            labels = torch.tensor(iteration[4])
            optimizer.zero_grad()
            output = model(jobduty_text, jobreq_text, workexp_text, projexp_text)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            StepLR.step()
            progbar.update(i + 1, None, None, [("train loss", loss.item())])

            if i == steps - 2:
                break

        print("epoch {} is over!\n".format(epoch))

    torch.save(model, model_save_path)
    print("\nTraining is over!\n")


if __name__ == "__main__":
    train_LM(config)
