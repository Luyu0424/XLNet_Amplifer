import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import random


def get_one_hot(labels):
    labels_one_hot = torch.zeros(len(labels), 15)
    for index, i in enumerate(labels):
        labels_one_hot[index][i] = 1
    return labels_one_hot


def get_cross_entropy_loss(inputs, labels, weight):
    # inputs : sen_num*class_num
    loss = torch.tensor([0])
    x_input = torch.softmax(inputs, dim=1)
    for i in range(x_input.size()[0]):
        for i, item in enumerate(zip(x_input[i], labels[i])):
            a, b = item
            if i == 1:
                loss = loss + b * torch.log(a)
            else:
                loss = loss + b * torch.log(a)

    loss = -loss / x_input.size()[0]
    return loss


def get_weight_cross_entropy_loss(inputs, labels, minority_list=None, weight=1.0):
    """
        inputs : sen_num*class_num
        labels : sen_num*class_num
        weight : number
    """
    loss = 0
    x_input = torch.softmax(inputs, dim=1)
    for i in range(x_input.size()[0]):
        for i, item in enumerate(zip(x_input[i], labels[i])):
            a, b = item
            if minority_list != None:
                if i in minority_list:
                    loss = loss + weight * b * torch.log(a)
                else:
                    loss = loss + b * torch.log(a)
            else:
                loss = loss + b * torch.log(a)

    loss = -loss / x_input.size()[0]
    return loss


def Iteration_cross_entropy_loss(inputs, labels, minority=None, weight=1.0):
    """
            inputs : batch*sen_num*class_num
            labels : sen_num*class_num
            weight : number
    """
    batch_num = inputs.size()[0]
    loss_total = 0
    for i in range(batch_num):
        if minority == None:
            loss_total += get_weight_cross_entropy_loss(inputs[i], labels)
        else:
            loss_total += get_weight_cross_entropy_loss(inputs[i], labels, minority, weight)
    return loss_total


def Iteration_cross_entropy_loss1(inputs, labels):
    """
            inputs : batch*sen_num*class_num
            labels : sen_num*class_num
            weight : number
    """
    batch_num = inputs.size()[0]
    loss_total = 0
    for i in range(batch_num):
        loss_total += F.cross_entropy(inputs[i],labels)
    return loss_total


if __name__ == '__main__':
    torch.manual_seed(1234)
    x_input = torch.randn(3, 3, 2)
    y_label = torch.tensor([1, 0, 0])
    y_target = torch.FloatTensor([[0, 1], [1, 0], [1, 0]])
    minority = torch.tensor([1])
    loss = Iteration_cross_entropy_loss(x_input, y_target, minority, 1)
    print(loss)
    loss1 = Iteration_cross_entropy_loss1(x_input, y_target)
    print(loss1)
