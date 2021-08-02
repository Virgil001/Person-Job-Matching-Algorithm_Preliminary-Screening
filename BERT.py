import numpy as np
import random
import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pylab as plt
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

# 设定超参数
SEED = 123
BATCH_SIZE = 16
learning_rate = 2e-5
weight_decay = 1e-2
epsilon = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Load and Preprocess Data
BASE_DIR = ''
DATA_PATH0 = os.path.join(BASE_DIR, 'data0710', '0填充后.csv')
DATA_PATH1 = os.path.join(BASE_DIR, 'data0710', '1填充后.csv')
df0 = pd.read_csv(DATA_PATH0, encoding='GB18030')
df1 = pd.read_csv(DATA_PATH1, encoding='GB18030')
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for i in range(df0.shape[0]):
    label_id = 0
    labels_index[0] = label_id
    a = ''
    for j in range(df0.loc[i].shape[0]):
        a += str(df0.loc[i][j])
    texts.append(a)
    labels.append(label_id)

for i in range(df1.shape[0]):
    label_id = 1
    labels_index[1] = label_id
    a = ''
    for j in range(df1.loc[i].shape[0]):
        a += str(df1.loc[i][j])
    texts.append(a)
    labels.append(label_id)
# print(texts[0], labels[0])
# print('Found %s texts.' % len(texts))
total_targets = torch.tensor(labels)
print(total_targets.shape)

# 用BertTokenizer进行编码
model_name = 'bert-base-chinese'
bert_path = './bert_model/'
# model_config = BertConfig.from_pretrained(model_name)
# model_config.output_hidden_states = True
# model_config.output_attentions = True
tokenizer = BertTokenizer.from_pretrained(model_name)


# model_name, cache_dir=cache_dir
# print(texts[2])
# print(tokenizer.encode(texts[2]))

# 将每一句转成数字 （大于1000做截断，小于1000做 Padding，加上首位两个标识，长度总共等于1002）
def convert_text_to_token(tokenizer, sentence, limit_size=510):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


input_ids = [convert_text_to_token(tokenizer, sen) for sen in texts]

input_tokens = torch.tensor(input_ids)
print(input_tokens[0])
print(input_tokens.shape)  # torch.Size([4573, 512])


# 建立mask
def attention_masks(input_ids):
    atten_mask = []
    for seq in input_ids:  # [4573, 512]
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_mask.append(seq_mask)
    return atten_mask


atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)
print(attention_tokens.shape)  # torch.Size([4573, 512])

# 划分训练集
from sklearn.model_selection import train_test_split

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens,
                                                                        total_targets,
                                                                        random_state=666,
                                                                        test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens,
                                                 random_state=666, test_size=0.2)
print(train_inputs.shape, test_inputs.shape)  # torch.Size([3658, 512]) torch.Size([915, 512])
print(train_masks.shape)  # torch.Size([3658, 512])和train_inputs形状一样

print(train_inputs[0])
print(train_masks[0])

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# 创建模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # num_labels表示2个分类,通过初审和不通过
device = torch.device('cpu')
model.to(device)
# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

epochs = 3
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)


# 定义评估函数
def binary_acc(preds, labels):  # preds.shape = [16, 2] labels.shape = [16, 1]
    # torch.max: [0]为最大值, [1]为最大值索引
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc


def tp_cnt(preds, labels):
    tp = 0
    for i in range(preds.size()[0]):
        if labels.flatten()[i] and torch.eq(torch.max(preds[i, :], dim=0)[1], labels.flatten()[i]):
            tp += 1
    return tp


def fp_cnt(preds, labels):
    fp = 0
    for i in range(preds.size()[0]):
        if labels.flatten()[i] and torch.ne(torch.max(preds[i, :], dim=0)[1], labels.flatten()[i]):
            fp += 1
    return fp


def fn_cnt(preds, labels):
    fn = 0
    for i in range(preds.size()[0]):
        if labels.flatten()[i] == 0 and torch.ne(torch.max(preds[i, :], dim=0)[1], labels.flatten()[i]):
            fn += 1
    return fn


def precision(tp_total, fp_total):
    p = tp_total / (tp_total + fp_total)
    return p


def recall(tp_total, fn_total):
    r = tp_total / (tp_total + fn_total)
    return r


def F_beta(tp_total, fp_total, fn_total):
    beta = 0.5
    p = precision(tp_total, fp_total)
    r = recall(tp_total, fn_total)
    f_beta = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r)
    return f_beta


# 计算模型运行时间
import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


# 训练函数
def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc = [], []
    tp_total, fp_total, fn_total = 0, 0, 0

    model.train()
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)

        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # criterion = nn.BCELoss()
        # logits = output[1]
        # loss = criterion(torch.max(logits, dim=1)[1].float(), b_labels.float())
        loss, logits = output[0], output[1]  # loss: 损失, logits: predict

        avg_loss.append(loss.item())

        acc = binary_acc(logits, b_labels)  # (predict, label)
        tp_total += tp_cnt(logits, b_labels)
        fp_total += fp_cnt(logits, b_labels)
        fn_total += fn_cnt(logits, b_labels)
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    r = recall(tp_total, fn_total)
    p = precision(tp_total, fp_total)
    fb = F_beta(tp_total, fp_total, fn_total)

    return avg_loss, avg_acc, r, p, fb


# 模型评估
def evaluate(model):
    avg_acc = []
    tp_total, fp_total, fn_total = 0, 0, 0

    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            acc = binary_acc(output[0], b_labels)
            tp_total += tp_cnt(output[0], b_labels)
            fp_total += fp_cnt(output[0], b_labels)
            fn_total += fn_cnt(output[0], b_labels)

            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    r = recall(tp_total, fn_total)
    p = precision(tp_total, fp_total)
    fb = F_beta(tp_total, fp_total, fn_total)
    return avg_acc, r, p, fb


# 训练模型并评估模型
for epoch in range(epochs):
    train_loss, train_acc, train_r, train_p, train_fb = train(model, optimizer)
    print('epoch={},Train Accuracy={}，Recall={}, Precision={}, F-Beta Score={}, Loss={}'.
          format(epoch + 1, train_acc, train_r, train_p, train_fb, train_loss))

    test_acc, test_r, test_p, test_fb = evaluate(model)
    print("epoch={}, Test Accuracy={}, Recall={}, Precision={}, F-Beta Score={}".
          format(epoch + 1, test_acc, test_r, test_p, test_fb))

# save the model
torch.save(model, './model.pkl')
