import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F
# a=torch.tensor([[0.1,2,1.3,0.3],[0.2,1.4,5,0.6]],dtype=float)
# print(torch.softmax(a,dim=0))
# print(torch.softmax(a,dim=1))
# embed()


class FeedForward(nn.Module):
    def __init__(self, hdim, out_dim):
        super(FeedForward, self).__init__()
        self.hidden_dim = hdim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.hidden_dim, self.out_dim)
        self.activate = nn.Sigmoid()

    def forward(self, input):
        output = input
        output = self.activate(self.linear1(output))
        return output


class Differ_Amplifier(nn.Module):
    def __init__(self, hidden_size, out_dim, layers):
        super(Differ_Amplifier, self).__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.layers = layers
        self.differ_layer = nn.ModuleList()
        for i in range(self.layers):
            self.differ_layer.append(nn.Linear(self.hidden_size, self.hidden_size, bias=False))

        self.FF = FeedForward(self.hidden_size, self.out_dim)

    def forward(self, input):
        #input:batch*paragraph_len*embed_dim
        H_v = input.squeeze(0)
        all_H = []
        for i in range(self.layers):
            F_v = self.differ_layer[i](self.get_represent(H_v))
            H_v = F_v+H_v
            all_H.append(H_v)
        stack_out = torch.stack(all_H)
        model_out = self.FF(stack_out)
        return model_out

    def get_represent(self, input):
        length, _ = input.size()
        F_x = []
        for i in range(length):
            all_i = torch.sum(input, dim=0)-input[i]
            out_i = input[i]-all_i/(length-1)
            F_x.append(out_i)
        out = torch.stack(F_x)
        return out


# model=Differ_Amplifier()
# input=torch.randn((1,5,768))
# outs=model(input)
#
#
# labels=torch.tensor([1,0,0,1,0])
# weight=(len(labels)-sum(labels))/sum(labels)
# labels_list=[]
# for i in labels:
#     if i==1:
#         labels_list.append(torch.tensor([0,weight]))
#     else:
#         labels_list.append(torch.tensor([1,0]))
# labels=torch.stack(labels_list)
# len,a,b=outs.size()
# loss_all=0
# for i in range(len):
#     loss=F.cross_entropy(outs[i],labels,reduction='mean')
#     loss_all+=loss
# print(loss_all)



