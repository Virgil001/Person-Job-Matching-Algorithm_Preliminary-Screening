import os
import torch
import numpy as np
import pandas as pd
import joblib
import config


GBDT_MODEL_SAVE_PATH = config.GBDT_SAVE_PATH
LANGUAGE_MODEL_SAVE_PATH = config.LM_SAVE_PATH
curdir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(curdir, "output.csv")


def get_test_data():
    data = pd.read_csv('./PreprocessedData.csv')
    data = data.fillna('无')
    test_df = data.sample(n=100, replace=True, random_state=None, axis=0)

    cols = [x for x in test_df.columns if x != '标签']

    test_features = test_df[cols]

    numeric_features_idx = test_features.dtypes[test_features.dtypes != 'object'].index
    numeric_features = test_features[numeric_features_idx]

    jobduty = test_features['clean_岗位职责']
    jobreq = test_features['clean_岗位要求']
    workexp = test_features['clean_工作经历']
    projexp = test_features['clean_项目经历']
    test_label = test_df['标签']

    return numeric_features, test_label, jobduty, jobreq, workexp, projexp


def predict(numeric_features, test_label, t1, t2, t3, t4, gbdt_model, language_model, OUTPUT_PATH):
    lgb_test = gbdt_model.Dataset(numeric_features.values, test_label.values, silent=True)
    gbdt_test_pre = gbdt_model.predict(lgb_test, num_iteration=gbdt_model.best_iteration)
    lm_test_pre = language_model(t1, t2, t3, t4)
    lm_test_pre.numpy().reshape(lgb_test.shape)
    pred = round((gbdt_test_pre + lm_test_pre) / 2)  
    result = pd.DataFrame(pred)
    result.columns = ['pred_results']
    result = pd.concat([result, test_label], axis=1)
    result.to_csv(OUTPUT_PATH, index=0, line_terminator="\r\r\n")


def joint_predictor():
    # torch.cuda.set_device(0)
    print("Start evluation...")
    numeric_features, test_label, t1, t2, t3, t4 = get_test_data()

    with torch.no_grad():
        gbdt_model = joblib.load(GBDT_MODEL_SAVE_PATH)
        language_model = torch.load(LANGUAGE_MODEL_SAVE_PATH)
        language_model()
        predict(numeric_features, test_label, t1, t2, t3, t4, gbdt_model, language_model, OUTPUT_PATH)

    print("Evaluation done! Result has saved to: ", OUTPUT_PATH)
