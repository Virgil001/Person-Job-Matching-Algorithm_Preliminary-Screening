import os
import torch
import numpy as np
import pandas as pd
import joblib
import config
import dataset as ds

GBDT_MODEL_SAVE_PATH = config.GBDT_SAVE_PATH
LANGUAGE_MODEL_SAVE_PATH = config.LM_SAVE_PATH
curdir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(curdir, "output.csv")
BATCH_SIZE = config.batch_size - 2


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

    gbdt_test_pre = gbdt_model.predict(numeric_features.values, num_iteration=gbdt_model.best_iteration)

    dataset_iterator = ds.get_iterator_batch(t1, t2, t3, t4, test_label)
    steps = ds.get_steps_per_epoch(len(test_label), BATCH_SIZE)
    lm_output = torch.tensor([])

    for i, iteration in enumerate(dataset_iterator):

        jobduty_text, jobreq_text, workexp_text, projexp_text = iteration[0], iteration[1], \
                                                                iteration[2], iteration[3]
        logits = language_model(jobduty_text, jobreq_text, workexp_text, projexp_text)

        lm_output = torch.cat((lm_output, logits), 0)

        if i == steps - 1:
            break
    lm_output = lm_output.numpy().reshape(len(test_label))

    pred = np.round((gbdt_test_pre + lm_output) / 2)  # 投票法 简单平均
    result = pd.DataFrame(pred)
    result.columns = ['pred_results']
    test_label = test_label.reset_index(drop=True)
    final_result = pd.concat([result, test_label], axis=1)
    final_result.to_csv(OUTPUT_PATH, index=0, line_terminator="\r\r\n")


def joint_predictor():
    # torch.cuda.set_device(0)
    print("Start evluation...")
    numeric_features, test_label, t1, t2, t3, t4 = get_test_data()

    with torch.no_grad():
        gbdt_model = joblib.load(GBDT_MODEL_SAVE_PATH)
        language_model = torch.load(LANGUAGE_MODEL_SAVE_PATH)
        predict(numeric_features, test_label, t1, t2, t3, t4, gbdt_model, language_model, OUTPUT_PATH)

    print("Evaluation done! Result has saved to: ", OUTPUT_PATH)


if __name__ == "__main__":
    joint_predictor()
