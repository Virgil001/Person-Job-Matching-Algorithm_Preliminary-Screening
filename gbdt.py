import pandas as pd
import numpy as np
import lightgbm as lgb
import evaluate as e
from sklearn.model_selection import StratifiedKFold
import dataset as ds
import joblib

EPOCH = ds.get_epoch()


def train_gbdt():
    for _ in range(EPOCH):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'max_bin': 50,
            'max_depth': 6,
            "learning_rate": 0.02,
            "colsample_bytree": 0.8,
            "bagging_fraction": 0.8,
            'min_child_samples': 25,
            'n_jobs': -1,
            'silent': True,
            'seed': 1000,
        }

        testresults = []

        kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=123)
        _, x, y, _, _, _, _ = ds.MyDataSet()
        _, test_x, test_y, _, _, _, _ = ds.MyDataSet()
        test_data = pd.concat([test_x, test_y], axis=1)

        for i, (train_index, valid_index) in enumerate(kf.split(x, y)):
            print("第", i + 1, "次")
            x_train, y_train = x.iloc[train_index], y.iloc[train_index]
            x_valid, y_valid = x.iloc[valid_index], y.iloc[valid_index]
            lgb_train = lgb.Dataset(x_train, y_train, silent=True)
            lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train, silent=True)
            gbm = lgb.train(params, lgb_train, num_boost_round=400, valid_sets=[lgb_train, lgb_eval], verbose_eval=100,
                            early_stopping_rounds=200)

            vaild_preds = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
            # 对测试集进行操作
            test_pre = gbm.predict(test_data.iloc[:, 1:], num_iteration=gbm.best_iteration)

            threshold = 0.45
            smalltestresults = []

            for w in test_pre:
                temp = 1 if w > threshold else 0
                smalltestresults.append(temp)
            testresults.append(smalltestresults)

            # 对每次交叉验证的验证集进行 0 ，1 化，然后评估f1值
            results = []
            for pred in vaild_preds:
                result = 1 if pred > threshold else 0
                results.append(result)
            c = e.F1_score(results, y_valid)
            print('F1值为：{}'.format(c))
        print('---N折交叉验证分数---')
        print(np.average(c))

    # 模型保存
    joblib.dump(gbm, "./gbdt_model.pkl")

# # 模型加载
# clf = joblib.load('dota_model.pkl')

# print(load_model.predict(test_x))

# 待调参


if __name__ == "__main__":
    train_gbdt()
