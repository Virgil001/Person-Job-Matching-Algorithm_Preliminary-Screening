from sklearn.metrics import f1_score, auc, confusion_matrix
import pandas as pd
import numpy as np


def F1_score(predict, real):
    score = []
    f1 = f1_score(real, predict, average='macro')
    score.append(f1)
    return score


def AUC(predict, real):
    score = []
    err = auc(real, predict)
    score.append(err)
    return score


def Confusion_M(predict, real):
    tn, fp, fn, tp = confusion_matrix(real, predict).ravel()
    return tn, fp, fn, tp


def show_results():
    result = pd.read_csv("./output.csv")
    pred_result, ground_truth = result['pred_result'].values, result['标签'].values
    f1 = np.average(F1_score(pred_result, ground_truth))
    print("The average F1 score is : {}".format(f1))
    auc = np.average(AUC(pred_result, ground_truth))
    print("The average AUC is : {}".format(auc))
    TN, FP, FN, TP = Confusion_M(ground_truth, pred_result)
    print("TN:{}".format(TN))
    print("FP:{}".format(FP))
    print("FN:{}".format(FN))
    print("TP:{}".format(TP))

