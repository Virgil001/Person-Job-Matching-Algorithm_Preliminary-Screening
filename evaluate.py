from sklearn.metrics import f1_score, auc, confusion_matrix
import pandas as pd
import numpy as np


def F1_score(predict, real):
    score = []
    f1 = f1_score(real, predict, average='macro')
    score.append(f1)
    return score


def Confusion_M(predict, real):
    tn, fp, fn, tp = confusion_matrix(real, predict).ravel()
    return tn, fp, fn, tp


def show_results():
    result = pd.read_csv("./output.csv")
    pred_result, ground_truth = result['pred_results'].values, result['标签'].values
    f1 = np.average(F1_score(pred_result, ground_truth))
    print("The average F1 score is : {}".format(f1))

    TN, FP, FN, TP = Confusion_M(ground_truth, pred_result)
    print("真反例TN:{}".format(TN))
    print("假正例FP:{}".format(FP))
    print("假反例FN:{}".format(FN))
    print("真正例TP:{}".format(TP))
    print("查准率:{}".format(TP / (TP + FP)))
    print("查全率:{}".format(TP / (TP + FN) if TP + FN != 0 else "inf"))


if __name__ == "__main__":
    show_results()


