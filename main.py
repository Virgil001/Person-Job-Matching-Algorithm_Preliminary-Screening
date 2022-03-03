import gbdt
import LanguageModel
import Joint_Predictor
import evaluate

if __name__ == "__main__":
    gbdt.train_gbdt()
    LanguageModel.train_LM()
    Joint_Predictor.joint_predictor()
    evaluate.show_results()
