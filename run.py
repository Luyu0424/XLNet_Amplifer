import datetime
import torch
import os
from Parameters import Args
from Model import Xlnet_Encoder_Amplifier
import numpy as np
import json
from IPython import embed
import random
import warnings
from Training import my_train, my_test, my_test1
warnings.filterwarnings('ignore')


def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(data_path, args):
    paragraphs, labels, para_spans, matrixs = [], [], [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            lin = json.loads(line)
            lab, para_span = [], []
            for item in lin['paragraph_lab']:
                lab.append(args.label_dict[item])

            for item in lin['non_leaf_node']:
                lab.append(args.label_dict[item[0]])
                para_span.append(item[1])
            paragraphs.append(lin['paragraphs'])
            labels.append(lab)
            para_spans.append(para_span)
            matrixs.append(lin['matrix'])
    # print(len(paragraphs), len(labels), len(para_spans))
    # print(paragraphs[0], '\n', labels[0], '\n', para_spans[0])
    return paragraphs, labels, para_spans, matrixs


def Parameters_Show(args):
    print('Seed_Num: {}'.format(args.seed_num))
    print('Drop_Rate: {}'.format(args.drop_rate))
    print('Training_Epoch: {}'.format(args.train_epoch))
    print('Learning_Rate: {}'.format(args.learning_rate))
    print('Encoder_Layer: {}'.format(args.encoder_layer))
    print('Encoder_Head: {}'.format(args.encoder_head))
    # print('GCN_Layer: {}'.format(args.gcn_layer))


def main():
    args = Args()
    Parameters_Show(args)
    set_seed(args.seed_num)
    train_data = get_data(args.train_data_path, args)
    test_data = get_data(args.test_data_path, args)
    model = Xlnet_Encoder_Amplifier(args).to(args.device)

    print('__________Training_start__________')
    train_start = datetime.datetime.now()
    #my_train(model, train_data, args)
    train_end = datetime.datetime.now()
    print('__________Training Cost Time: {}__________'.format(train_end-train_start))
    print('\n'*4)
    print('__________Testing_start__________')
    test_start = datetime.datetime.now()
    my_test1(model, test_data, args)
    test_end = datetime.datetime.now()
    print('__________Testing Cost Time: {}__________'.format(test_end - test_start))


if __name__ == '__main__':
    main()
    # args = Args()
    # train_data = get_data(args.train_data_path, args)
    # # train_data_add = get_data(args.train_data_add_path, args)
    # test_data = get_data(args.test_data_path, args)
    # all_para = train_data[1]+test_data[1]
    # dicts = {}
    # for i in range(17):
    #     dicts.update({i: 0})
    # all_num = 0
    # for para in all_para:
    #     for ele in para:
    #         dicts[ele] += 1
    #         all_num += 1
    # print(dicts)
    # for i in dicts:
    #     dicts[i]/=all_num
    # print(dicts)
