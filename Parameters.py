import os
import torch


class Args:
    def __init__(self):
        if os.path.exists("./save_parameters"):
            pass
        else:
            os.mkdir("./save_parameters")
        if os.path.exists("./results"):
            pass
        else:
            os.mkdir("./results")
        if os.path.exists('../xlnet_base'):
            self.checkpoint = '../xlnet_base'
        else:
            self.checkpoint = 'F:/myproject/xlnet_base'
        self.saved_model_path = './save_parameters/'
        self.train_data_path = './datas/train_all.jsonl'
        self.train_data_add_path = './datas/train_add_all.jsonl'
        self.test_data_path = './datas/test_all.jsonl'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = 768  # 词嵌入维度
        self.hidden_size = 768  # 隐藏层维度
        self.drop_rate = 0.1  # drop
        self.learning_rate = 1e-5  # 学习率
        self.warmup_proportion = 0.1
        self.train_epoch = 50  # 训练轮数
        self.vocab_size = 32000  # 词表大小
        self.adam_epsilon = 1e-8
        self.encoder_head = 8  # encoder多头注意力的头数
        self.encoder_layer = 12  # encoder的层数
        self.gcn_layer = 2
        self.Amplifier_layer = 2
        self.seed_num = 38

        self.target_class = ['Background', 'Behavior', 'Cause', 'Comment', 'Contrast', 'Illustration', 'Lead',
                             'Progression', 'Purpose', 'Result', 'Situation', 'Statement', 'Story', 'Sub-Summary',
                             'Summary', 'Sumup', 'Supplement']
        self.target_class_18 = ['Background', 'Behavior', 'Cause', 'Comment', 'Contrast', 'Illustration', 'Lead',
                             'Progression', 'Purpose', 'Result', 'Situation', 'Statement', 'Story', 'Sub-Summary',
                             'Summary', 'Sumup', 'Supplement', 'News']
        self.class_num = len(self.target_class)  # 输出类数量
        self.label_dict = {
            'Background': 0,
            'Behavior': 1,
            'Cause': 2,
            'Comment': 3,
            'Contrast': 4,
            'Illustration': 5,
            'Lead': 6,
            'Progression': 7,
            'Purpose': 8,
            'Result': 9,
            'Situation': 10,
            'Statement': 11,
            'Story': 12,
            'Story-Situation': 10,
            'Sub-Summary': 13,
            'Summary': 14,
            'Summary-Lead': 6,
            'Sumup': 15,
            'Supplement': 16
        }
        self.minority_class = [1, 4, 5, 7, 8, 11, 15]  # <3%
        self.majority_class = [6, 10, 12, 16]  # >8%



        # self.target_class = ['Background', 'Behavior', 'Cause', 'Comment', 'Contrast', 'Illustration', 'Lead',
        #                      'Progression', 'Purpose', 'Result', 'Situation', 'Statement', 'Sub-Summary',
        #                      'Sumup', 'Supplement']
        # self.class_num = len(self.target_class)  # 输出类数量
        # self.minority_class = [1, 4, 5, 7, 8, 11, 13]
        # self.majority_class = [6, 10, 12, 14]
        # self.label_dict = {
        #     'Background': 0,
        #     'Behavior': 1,
        #     'Cause': 2,
        #     'Comment': 3,
        #     'Contrast': 4,
        #     'Illustration': 5,
        #     'Lead': 6,
        #     'Progression': 7,
        #     'Purpose': 8,
        #     'Result': 9,
        #     'Situation': 10,
        #     'Statement': 11,
        #     'Story-Situation': 10,
        #     'Sub-Summary': 12,
        #     'Summary-Lead': 6,
        #     'Sumup': 13,
        #     'Supplement': 14
        # }