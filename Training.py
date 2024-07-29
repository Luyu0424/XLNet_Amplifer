import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import datetime
from IPython import embed
from Loss_function import *


minority_class = [1, 4, 5, 7, 8, 11, 15]  # <3%
majority_class = [6, 10, 12, 16]  # >8%

# minority_class = [1, 4, 5, 7, 8, 11, 13]
# majority_class = [6, 10, 12, 14]


def get_weight(labels):
    minor, mosts = 0, 0
    for labb in labels:
        if labb in minority_class:
            minor += 1
        elif labb in majority_class:
            mosts += 1
    weig = 2
    if minor != 0:
        weig = mosts/minor
    weig = max(weig, 1)
    return weig


def my_train(model, train_data, args):
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = lr.MultiStepLR(optimizer, milestones=[15000, 25000], gamma=0.1)
    for epoch in range(args.train_epoch):
        torch.cuda.empty_cache()
        print('***' * 5 + 'epoch_{} start:'.format(epoch) + '***' * 10)
        start_time = datetime.datetime.now()
        num_iter = 0
        predict_batch = np.array([], dtype=int)
        labels_batch = np.array([], dtype=int)
        loss_batch = 0
        for item in zip(train_data[0], train_data[1], train_data[2]):
            model.train()
            sentence = item[0]
            label = torch.tensor(item[1]).to(args.device)
            label1 = label[:len(sentence)]
            label2 = label[len(sentence):]
            para_span = item[2]
            class_out, class_out1 = model(sentence, para_span)
            true = label.data.cpu()
            weight = get_weight(true)
            one_hot_label = get_one_hot(label1)

            loss1 = Iteration_cross_entropy_loss(class_out, one_hot_label, minority_class, majority_class, weight)
            if class_out1!=None:
                loss2 = F.cross_entropy(class_out1, label2)
            else:
                loss2 = 0
            loss = loss1 + loss2
            # loss = F.cross_entropy(class_out, label)
            labels_batch = np.append(labels_batch, item[1])
            predict1 = torch.max(class_out[-1].data, 1)[1].cpu()
            predict_batch = np.append(predict_batch, predict1)
            if class_out1!=None:
                predict2 = torch.max(class_out1.data, 1)[1].cpu()
                predict_batch = np.append(predict_batch, predict2)
            loss_batch += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (num_iter + 1) % 20 == 0:
                train_acc = metrics.accuracy_score(labels_batch, predict_batch)
                print('Iter:{iter},train_loss:{t_l:.2} , train_acc:{t_a:.2}'.format(
                    iter=num_iter + 1, t_l=loss_batch / 20, t_a=train_acc))
                predict_batch = np.array([], dtype=int)
                labels_batch = np.array([], dtype=int)
                loss_batch = 0
            num_iter += 1
        if epoch >= 5:
            torch.save(model.state_dict(), args.saved_model_path + 'epoch_{}.ckpt'.format(epoch))
        end_time = datetime.datetime.now()
        print('one epoch cost:', end_time - start_time)
        print('***' * 5 + 'epoch_{} end:'.format(epoch) + '***' * 10)
        print('\n' * 2)


def my_test(model, test_data, args):
    for epoch in range(5, args.train_epoch):
        model.load_state_dict(torch.load(args.saved_model_path + 'epoch_{}.ckpt'.format(epoch)))
        print('__'*5+'Testing epoch_{}'.format(epoch)+'__'*5)
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in zip(test_data[0], test_data[1], test_data[2], test_data[3]):
                sentence = item[0]
                label = torch.tensor(item[1]).to(args.device)
                label1 = label[:len(sentence)]
                label2 = label[len(sentence):]
                para_span = item[2]
                class_out, class_out1 = model(sentence, para_span)
                if class_out1!=None:
                    loss = F.cross_entropy(class_out[-1], label1) + F.cross_entropy(class_out1, label2)
                else:
                    loss = F.cross_entropy(class_out[-1], label1)
                labels_all = np.append(labels_all, item[1])
                predict1 = torch.max(class_out[-1].data, 1)[1].cpu()
                if class_out1!=None:
                    predict2 = torch.max(class_out1.data, 1)[1].cpu()
                predict_all = np.append(predict_all, predict1)
                if class_out1 != None:
                    predict_all = np.append(predict_all, predict2)
                loss_total += loss



        acc = metrics.accuracy_score(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=args.target_class, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        ave_loss = loss_total / len(test_data[0])
        msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
        print(msg.format(ave_loss, acc))
        print('Precision, Recall and F1-Score')
        print(report)
        print('Confusion Maxtrix')
        print(confusion)
        print('_' * 20)
        print('\n' * 2)


def my_test1(model, test_data, args):
    for epoch in range(5, args.train_epoch):
        model.load_state_dict(torch.load(args.saved_model_path + 'epoch_{}.ckpt'.format(epoch)))
        print('__'*5+'Testing epoch_{}'.format(epoch)+'__'*5)
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in zip(test_data[0], test_data[1], test_data[2], test_data[3]):
                sentence = item[0]
                label = torch.tensor(item[1]).to(args.device)
                label1 = label[:len(sentence)]
                label2 = label[len(sentence):]
                para_span = item[2]
                class_out, class_out1 = model(sentence, para_span)
                if class_out1!=None:
                    loss = F.cross_entropy(class_out[-1], label1) + F.cross_entropy(class_out1, label2)
                else:
                    loss = F.cross_entropy(class_out[-1], label1)
                labels_all = np.append(labels_all, item[1])
                labels_all = np.append(labels_all, 17)
                predict1 = torch.max(class_out[-1].data, 1)[1].cpu()
                if class_out1!=None:
                    predict2 = torch.max(class_out1.data, 1)[1].cpu()
                predict_all = np.append(predict_all, predict1)
                if class_out1 != None:
                    predict_all = np.append(predict_all, predict2)
                predict_all = np.append(predict_all, 17)
                loss_total += loss



        acc = metrics.accuracy_score(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=args.target_class_18, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        ave_loss = loss_total / len(test_data[0])
        msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
        print(msg.format(ave_loss, acc))
        print('Precision, Recall and F1-Score')
        print(report)
        print('Confusion Maxtrix')
        print(confusion)
        print('_' * 20)
        print('\n' * 2)
