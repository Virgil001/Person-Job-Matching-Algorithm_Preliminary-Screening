from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QTextEdit, QLabel, QComboBox, QLineEdit
from PyQt5.QtGui import QIcon
import sys
import torch
from transformers import BertTokenizer


def convert_text_to_token(tokenizer, sentence, limit_size=510):
    tokens = tokenizer.encode(sentence[:limit_size])
    if len(tokens) < limit_size + 2:
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


def predict(sen):
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device('cpu')
    input_id = convert_text_to_token(tokenizer, sen)
    input_token = torch.tensor(input_id).long().to(device)

    atten_mask = [float(i > 0) for i in input_id]
    attention_token = torch.tensor(atten_mask).long().to(device)

    output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_token.view(1, -1))
    print(output[0])

    return torch.max(output[0], dim=1)[1]


class person_job_match_app(QWidget):
    def __init__(self):
        super().__init__()

        self.name_l = QLabel("姓名", self)
        self.name_l.move(200, 25)
        self.name = QLineEdit(self)
        self.name.move(200, 50)
        self.name.setPlaceholderText("请输入您的姓名")

        self.name_l = QLabel("性别", self)
        self.name_l.move(500, 25)
        self.name = QLineEdit(self)
        self.name.move(500, 50)
        self.name.setPlaceholderText("性别")

        self.name_l = QLabel("年龄", self)
        self.name_l.move(800, 25)
        self.name = QLineEdit(self)
        self.name.move(800, 50)
        self.name.setPlaceholderText("年龄")

        self.combo = QComboBox(self)
        self.lbl = QLabel("申请岗位", self)
        self.a6 = QTextEdit(self)
        self.a5 = QLabel(self)
        self.a4 = QTextEdit(self)
        self.a3 = QLabel(self)
        self.a2 = QTextEdit(self)
        self.a1 = QLabel(self)
        self.btn = QPushButton('提交', self)
        self.btn.move(900 * 2, 610 * 2)
        self.btn.clicked.connect(self.Judge)
        self.re_label = QLabel("初审结果：", self)
        self.re_label.move(455 * 2, 633 * 2)
        self.result = QLineEdit(self)
        self.initUI()
        self.Judge()

    def initUI(self):
        self.setGeometry(80, 45, 1000 * 2, 660 * 2)
        self.setWindowTitle('SHU智联招聘v1.0.0')
        self.setWindowIcon(QIcon('shu.png'))

        self.lbl.move(200, 85)
        self.combo.addItem("请选择")
        self.combo.addItem("大数据")
        self.combo.addItem("Java工程师")
        self.combo.move(200, 110)

        self.a1.setText('* 请简要描述您的工作经历（不超过200字）')
        self.a1.move(200, 150)
        self.a2.move(200, 175)
        self.a2.resize(1024, 300)

        self.a3.setText('* 请简要描述您的项目经历（不超过200字）')
        self.a3.move(200, 495)
        self.a4.move(200, 525)
        self.a4.resize(1024, 300)

        self.a5.setText('* 请简述您的个人优势（不超过100字）')
        self.a5.move(200, 850)
        self.a6.move(200, 875)
        self.a6.resize(1024, 300)

        self.result.move(500 * 2, 630 * 2)
        self.result.setStyleSheet("color:red")
        self.show()  # 显示窗口

    def Judge(self):
        texts = [self.a2.toPlainText(),
                 self.a4.toPlainText(),
                 self.a6.toPlainText()]
        if self.combo.currentText() == "Java工程师":
            texts.append("1、专科及以上学历，计算机相关专业，1-3年JAVA开发经验；"
                         "2、JAVA基础扎实，熟悉主流开源应用框架，如Spring、MyBatis、XML、JSON、Maven等技术；"
                         "3、熟悉并理解常用中间件服务，如消息队列、缓存、存储等；"
                         "4、掌握常用的数据结构、设计模式，能解决实际问题者优先；"
                         "5、掌握数据库原理和知识，能独立DB设计、SQL开发和性能调优；"
                         )
        else:
            texts = None

        if not texts:
            self.result.setText(" ")
        else:
            label = predict(texts)
            if label == 1:
                self.result.setText("恭喜，初审通过！")
            else:
                self.result.setText("初审未通过，还望再接再厉！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = torch.load('./model.pkl')
    w = person_job_match_app()
    sys.exit(app.exec_())
