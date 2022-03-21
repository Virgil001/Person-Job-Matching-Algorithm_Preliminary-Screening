import gbdt
import LanguageModel
import Joint_Predictor
import evaluate
import config

if __name__ == "__main__":
    gbdt.train_gbdt()
    LanguageModel.train_LM(config)
    Joint_Predictor.joint_predictor()
    evaluate.show_results()
