"""
NoisyWikihow DataSet & Models

Note that some models only adapt to Bart, since different pretrained models differ in many aspects,
e.g: config, pad_token_id, initialization strategies, classification heads.

Any Model can run 'base' method correctly.
Bart can run every methods.
"""


import os
from torch.utils.data import DataLoader, Dataset
from cmd_args import args, Wiki_CONFIG
import torch.nn as nn
import torch

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, DataCollatorWithPadding
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification
from transformers import BartConfig, BartTokenizer, BartForSequenceClassification
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset

from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.nn import Module

from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers.models.bart.modeling_bart import BartClassificationHead



MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "bart": (BartConfig, BartForSequenceClassification, BartTokenizer),
    "gpt2": (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}


tokenizer_list = {
    "bert": args.model_path["bert"],
    "xlnet": args.model_path["xlnet"],
    "roberta": args.model_path["roberta"],
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "gpt2": args.model_path["gpt2"],
    "t5": args.model_path["t5"],
}

print("Loading the tokenizer")

# Use 2-layer bidrectional LSTM to generate the sentence representation
class LSTM(nn.Module):
    def __init__(self, embedding, num_layers=2, bidrectional=True):
        super().__init__()
        num_embeddings, embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.embedding = embedding # load embedding from pretrained model
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = embedding_dim, num_layers = num_layers, batch_first = True, bidirectional = bidrectional)
        self.bidrectional = bidrectional
    # Use inputs_embeds first
    def forward(self,input_ids=None,inputs_embeds=None,attention_mask=None,**kwargs):
        if attention_mask is not None:
            length = attention_mask.sum(-1).cpu()
        else:
            # TODO Bart: padding_idx=1, models may differ in pad_token_id
            length = torch.tensor([ids.not_equal(1).sum() for ids in input_ids]).cpu()
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Neither input_ids or inputs_embeds was passed")
            inputs_embeds = self.embedding(input_ids)
        pack_embed = pack_padded_sequence(inputs_embeds, length, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(pack_embed)
        if self.bidrectional:
            # (batch_size,2*embedding_dim)
            return torch.cat([h[-2],h[-1]],dim=-1)
        else:
            # (batch_size, embedding_dim)
            return h[-1]
    def get_input_embeddings(self):
        return self.embedding

# retain the embedding and classification_head structure For Bart
class LSTMForSequenceClassification(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        config: BartConfig = model.config
        self.config = config
        self.model = LSTM(model.get_input_embeddings(), **kwargs)
        # the old model embedding may changed when training
        self.bidrectional = self.model.bidrectional
        lstm_frac = 2 if self.bidrectional else 1
        self.classification_head = BartClassificationHead(
            config.d_model * lstm_frac,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        # init weights (Bart Model)
        self._init_weights(self.classification_head.dense)
        self._init_weights(self.classification_head.out_proj)
    # TODO Different model may have different initialization strategies.
    #  This method was based on Bart Model.
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    def forward(self,**kwargs):
        sentence_representation = self.model(**kwargs)
        logits = self.classification_head(sentence_representation)
        return Seq2SeqSequenceClassifierOutput(logits=logits)
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

# Update
class BartCLSWithEmbeds(Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.old_model = model
        self.bart = model.model
        self.classification_head = model.classification_head
    
    # pass input_ids to find the [EOS] token, whose decoder outputs represent the sentence
    def forward(self,input_ids,inputs_embeds=None,decoder_inputs_embeds=None,attention_mask=None):

        # No inputs_embeds. (e.g. Test)
        if inputs_embeds is None:
            return self.old_model(input_ids=input_ids, attention_mask=attention_mask)

        # Those model can handle inputs_embeds directly
        if isinstance(self.old_model, LSTMForSequenceClassification):
            return self.old_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, attention_mask=attention_mask)
        

        bart_outputs = self.bart(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, attention_mask=attention_mask)
        hidden_states = bart_outputs[0]
        # Use Eos token
        # TODO Note that GPT2 doesn't set pad_token_id, if you set 'pad_token_id==eos_token_id',
        #  the following steps won't match GPT2 
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)
        return Seq2SeqSequenceClassifierOutput(logits=logits) # 输出形式统一一下
    
    def get_input_embeddings(self):
        return self.bart.get_input_embeddings()

    # prepare inputs_embeds & decoder_inputs_embeds
    # ref: https://github.com/huggingface/transformers/issues/9388#issuecomment-753630982
    def generate_embeds(self, input_ids):
        decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
        return inputs_embeds, decoder_inputs_embeds


'''
Data Format: x,y,index
Noise on feature: train(x_noisy, y) test(x,y) e.g: mix, tail, uncommon, neighbor
Noise on label: train(x, y_noisy) test(x,y)   e.g: sym, idn
'''
class WikiDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y)
    # x, y, index
    def __getitem__(self, index):
        item = {k:v[index] for k,v in self.x.items()} 
        return item, self.y[index], index
    def __len__(self):
        return len(self.y)

class WikiDataSetSoft(Dataset):
    def __init__(self, wikiset, y_soft):
        self.x = wikiset.x
        self.y = wikiset.y
        self.y_soft = y_soft
    # x, y, y_soft, index
    def __getitem__(self, index):
        item = {k:v[index] for k,v in self.x.items()} 
        return item, self.y[index], self.y_soft[index], index
    def __len__(self):
        return len(self.y)

def get_wiki_train_and_val_loader(args):
    print('==> Preparing data for sst..')
    test_csv = pd.read_csv(f"{args.data_path}/noisy/test.csv")
    data = torch.load(f"{args.data_path}/embedding/{args.model_type}.pt")
    test_step = test_csv["step_id"].to_list()
    test_step = {k:v[test_step] for k,v in data.items()}
    test_cat = test_csv["cat_id"].to_list()

    if args.noise_rate!=0:
        train_csv = pd.read_csv(f"{args.data_path}/noisy/{args.noise_mode}_{args.noise_rate:.1f}.csv")
        train_step = train_csv["step_id"].to_list()
        train_cat = train_csv["cat_id"].to_list() 
        # Noise on feature:  train(x_noisy, y)
        if 'noisy_id' in train_csv:
            train_noisy = train_csv["noisy_id"].to_list()
            noisy_ind = [i for i in range(len(train_step)) if train_step[i]!=train_noisy[i]]
            clean_ind = [i for i in range(len(train_step)) if train_step[i]==train_noisy[i]]
            train_noisy = {k:v[train_noisy] for k,v in data.items()}

            trainset = WikiDataSet(train_noisy, train_cat)

        # Noise on label:  train(x, y_noisy)
        else:
            noisy_y = train_csv["noisy_label"].to_list()
            noisy_ind = [i for i in range(len(train_step)) if train_cat[i]!=noisy_y[i]]
            clean_ind = [i for i in range(len(train_step)) if train_cat[i]==noisy_y[i]]
            train_step = {k:v[train_step] for k,v in data.items()}
            
            trainset = WikiDataSet(train_step, noisy_y)

    else:
        train_csv = pd.read_csv(f"{args.data_path}/noisy/train.csv")
        train_step = train_csv["step_id"].to_list()
        noisy_ind = []
        clean_ind = list(range(len(train_step)))
        train_step = {k:v[train_step] for k,v in data.items()}
        train_cat = train_csv["cat_id"].to_list()

        trainset = WikiDataSet(train_step, train_cat)

    
    testset = WikiDataSet(test_step, test_cat)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, test_loader, noisy_ind, clean_ind

def get_wiki_model_and_loss_criterion(args):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer = tokenizer_class.from_pretrained(args.model_path[args.model_type], do_lower_case=True)
    config = config_class.from_pretrained(args.model_path[args.model_type], num_labels = args.num_class)
    ##############
    print('Loading {}'.format(args.model_type))

    # TODO Note: GPT2 didn't set the pad_token, we can set 'pad_token_id = eos_token_id'
    # https://github.com/huggingface/transformers/issues/3021
    # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
    if args.model_type == 'gpt2':
        config.pad_token_id = config.eos_token_id

    model = model_class.from_pretrained(
        args.model_path[args.model_type],
        config=config,
    )
    model.to(args.device)
    if args.use_lstm:
        print('use lstm, embeddings init from {}'.format(args.model_type))
        model = LSTMForSequenceClassification(model).to(args.device)
    criterion = Wiki_CONFIG[args.loss].to(args.device)
    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)

    return model, criterion, criterion_val

def get_wiki_tokenizer_and_label(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_path[args.model_type], do_lower_case=True)
    cat = pd.read_csv(f'{args.data_path}/cat158.csv')['category'].map(lambda x:x.lower()).to_list() # 全小写
    cat_token = tokenizer(cat, padding='max_length', max_length=15, truncation=True, return_tensors='pt')['input_ids'].to(args.device)
    cat_labels = tokenizer.batch_decode(cat_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    cat_token[cat_token==tokenizer.pad_token_id]=-100
    return tokenizer, cat_token, cat_labels

def save_index():
    import os
    import json
    index_path = 'corrupt_index'
    os.makedirs(index_path,exist_ok=True)
    for mode in ['mix','sym','idn']:
        args.noise_mode = mode
        print(mode)
        home = index_path
        for nr in [0.1, 0.2, 0.4, 0.6]:
            args.noise_rate = nr
            train_loader, test_loader, noisy_ind, clean_ind = get_wiki_train_and_val_loader(args)
            with open(os.path.join(home,f'{mode}_{nr}_noisy.txt'),"w") as f:
                json.dump(noisy_ind, f)
            print(f"nr{nr}: noisy{len(noisy_ind)}\tclean{len(clean_ind)}")


if __name__ == '__main__':
    pass

