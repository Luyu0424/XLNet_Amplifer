from Encoder import Encoder
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from differ_amp import Differ_Amplifier


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.hidden_size = hidden_size
        #Linear
        self.W_d = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.U_d = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(self, input1, input2):
        #Linear
        G_d = torch.sigmoid_(self.W_d(input1) + self.U_d(input2))
        v_d = torch.mul(G_d, input1) + torch.mul(1-G_d, input2)
        return  v_d


class Xlnet_Encoder_Amplifier(nn.Module):
    def __init__(self, args):
        super(Xlnet_Encoder_Amplifier, self).__init__()
        self.device = args.device
        self.xlnet = AutoModel.from_pretrained(args.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        self.encode = Encoder(device=self.device, n_layers=args.encoder_layer, d__model=args.embed_dim,
                              d_k=args.embed_dim, d_v=args.embed_dim, h=args.encoder_head)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.DiffAmp = Differ_Amplifier(args.hidden_size, args.class_num, args.Amplifier_layer)
        # self.gate_control = Gate(hidden_size=args.hidden_size)
        self.classifier = nn.Linear(args.embed_dim, args.class_num)

    def forward(self, sentence, para_span):
        # sen_vec_global = self.get_xlnet_represent_global(sentence).squeeze(0)  # [sen_num, embed_dim]
        # sen_vec_local = self.get_xlnet_represent_local(sentence)  # [sen_num, embed_dim]
        # sen_vec = self.gate_control(sen_vec_global, sen_vec_local).unsqueeze(0)
        sen_vec = self.get_xlnet_represent_global(sentence)
        encoder_input = self.dropout(sen_vec)
        out = self.dropout(self.encode(encoder_input))  # [batch,sen_num,embed_dim]
        # class_out = self.DiffAmp(out) # [sen_num,class_num]

        combine_embeddings = out.squeeze(0)
        span_embeddings = []
        for span in para_span:
            span_embeddings.append(torch.mean(combine_embeddings[span[0]-1:span[1], :], dim=0))   # span段落的开始为1，但向量开始为0，所以要 -1
        # if span_embeddings != []:
        #     span_embeddings = torch.stack(span_embeddings)
        #     all_embeddings = torch.cat((combine_embeddings, span_embeddings), dim=0)
        # else:
        #     all_embeddings = combine_embeddings
        class_out1 = self.DiffAmp(out)            # [sen_num,class_num]
        if span_embeddings!=[]:
            span_embeddings = torch.stack(span_embeddings)
            class_out2 = self.classifier(span_embeddings)
            # print(class_out.shape)
            # class_out = self.classifier(all_embeddings)            # [sen_num,class_num]
            return class_out1, class_out2
        else:
            return class_out1, None

    def get_xlnet_represent1(self, content):
        output, sep_index = [], []
        for sentence in content:
            token = self.tokenizer.tokenize(sentence)
            token = token + ['<sep>']
            output += token
            sep_index.append(len(output) - 1)
        output.append('<cls>')
        token_id = self.tokenizer.convert_tokens_to_ids(output)
        token_tensor = torch.tensor([token_id]).to(self.device)
        embedding = self.xlnet(token_tensor)[0]
        return embedding, sep_index

    def get_xlnet_represent_global(self, content):
        output, sep_index = [], []
        for sen in content:
            token = self.tokenizer.tokenize(sen) + ['<sep>']
            output += token
            sep_index.append(len(output) - 1)
        output.append('<cls>')
        token_id = self.tokenizer.convert_tokens_to_ids(output)
        token_tensor = torch.tensor([token_id]).to(self.device)
        embedding = self.xlnet(token_tensor)[0]
        sen_vec = embedding[:, sep_index, :]
        return sen_vec

    def get_xlnet_represent_local(self, content):
        output = []
        for sen in content:
            token = self.tokenizer.tokenize(sen) + ['<sep>', '<cls>']
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            token_tensor = torch.tensor([token_id]).to(self.device)
            embedding = self.xlnet(token_tensor)[0].squeeze(0)
            output.append(embedding[-1])

        output = torch.stack(output)
        return output
