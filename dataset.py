import pandas as pd
import random
from sklearn.utils import shuffle
import config


def get_epoch():
    data = pd.read_csv('./PreprocessedData.csv')
    data = data.fillna('无')
    dataSet = list(data.groupby('标签'))
    return len(dataSet[0][1]) // len(dataSet[1][1])


def get_max_seq_len():
    data = pd.read_csv('./PreprocessedData.csv')
    data = data.fillna('无')

    text_idx = data.dtypes[data.dtypes == 'object'].index
    text_data = data[text_idx]

    jobduty = text_data['clean_岗位职责']
    jobreq = text_data['clean_岗位要求']
    workexp = text_data['clean_工作经历']
    projexp = text_data['clean_项目经历']

    seq_lens = []

    for i in jobduty.index:
        seq_lens.append(len(jobduty.loc[i]))
    for i in jobreq.index:
        seq_lens.append(len(jobreq.loc[i]))
    for i in workexp.index:
        seq_lens.append(len(workexp.loc[i]))
    for i in projexp.index:
        seq_lens.append(len(projexp.loc[i]))

    max_seq_len = max(seq_lens)

    return max_seq_len


def MyDataSet():
    data = pd.read_csv('./PreprocessedData.csv')
    data = data.fillna('无')
    dataSet = list(data.groupby('标签'))

    F_sample = dataSet[0][1].sample(n=91, replace=True, random_state=None, axis=0)  # 91为数据集中的正样本数
    P_sample = dataSet[1][1]

    train_data = pd.concat([F_sample, P_sample], axis=0)
    train_data = shuffle(train_data)
    cols = [x for x in train_data.columns if x != '标签']
    train_features = train_data[cols]  # .values
    numeric_features_idx = train_features.dtypes[train_features.dtypes != 'object'].index
    numeric_features = train_features[numeric_features_idx]
    jobduty = train_features['clean_岗位职责']
    jobreq = train_features['clean_岗位要求']
    workexp = train_features['clean_工作经历']
    projexp = train_features['clean_项目经历']
    train_label = train_data['标签']

    return len(train_label), numeric_features, train_label, jobduty, jobreq, workexp, projexp


def data_clean(data):
    # <class 'Series'>
    dic = []
    s = pd.Series(dic)
    # <class 'str'>
    c = '1'
    # <class 'numpy.int64'>
    num = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
    aa = num['A']
    if type(data) == type(s):
        return data.values[0]
    elif type(data) == type(c):
        return data
    elif type(data) == type(aa[1]):
        return int(data)


def get_iterator_batch(jobduty, jobreq, workexp, projexp, train_label, batch_size=config.batch_size):
    def get_text_and_label_index_iterator(jobduty, jobreq, workexp, projexp, train_label):
        for idx in jobduty.index:
            jobduty_text = data_clean(jobduty.loc[idx])
            jobreq_text = data_clean(jobreq.loc[idx])
            workexp_text = data_clean(workexp.loc[idx])
            projexp_text = data_clean(projexp.loc[idx])
            label = data_clean(train_label.loc[idx])

            yield jobduty_text, jobreq_text, workexp_text, projexp_text, label

    data_iter = get_text_and_label_index_iterator(jobduty, jobreq, workexp, projexp, train_label)
    continue_iterator = True
    while continue_iterator:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(data_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False
        random.shuffle(data_list)

        jd_t_list, jr_t_list, we_t_list, pe_t_list = [], [], [], []
        label_list = []

        for data in data_list:
            jd_t, jr_t, we_t, pe_t, label = data[0], data[1], data[2], data[3], data[4]

            jd_t_list.append(jd_t)
            jr_t_list.append(jr_t)
            we_t_list.append(we_t)
            pe_t_list.append(pe_t)
            label_list.append(label)

        yield jd_t_list, jr_t_list, we_t_list, pe_t_list, label_list

    return False
