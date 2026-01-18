import json
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm
import re
from torch import cuda
from enum import Enum
from datasets import load_dataset
import logging
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

class TokenMultiTaskModel(torch.nn.Module):
    def __init__(self, llm_name, num_labels, max_length=128):
        super(TokenMultiTaskModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(llm_name, attn_implementation="eager")
        self.base_model.base_model.base_model.config.output_attentions = True
        # self.base_model.resize_token_embeddings(len_tok)
        self.num_labels = num_labels
        self.max_length = max_length
        self.d_model = self.base_model.config.hidden_size # modificare se richiesto
        # self.d_model = self.base_model.config.d_model
        self.dropout = torch.nn.Dropout(0.25)
        self.linear_sentence = torch.nn.Linear(self.d_model, 1) 
        self.linear_token = torch.nn.Linear(self.d_model, 1) 
        self.sigmoid = torch.nn.Sigmoid()
        
    
    def forward(self, input_ids, attention_mask):
        # output = self.base_model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        attention = output.attentions[-1].mean(dim=1) # (batch_size, seq_len, seq_len)
        
        output = output.last_hidden_state # (batch_size, seq_lenn, hidden_size)
        output_cls = output[:, 0, :] # (batch_size, hidden_size)
        cls_attention = attention[:, 0, :] # (batch_size, seq_len)

        # calcolo se la frase é fallace o meno
        output_cls = self.dropout(output_cls)
        logits_sentence = self.linear_sentence(output_cls)
  
        output = self.dropout(output)
        logits_token = self.linear_token(output)
        
        return logits_sentence, logits_token, cls_attention
def use_multi_task_classification_model(text, model, tokenizer, th, device):
    # tokenizzare il testo
    input_ids = tokenizer(text, return_tensors='pt', max_length= 256, padding='max_length', truncation=True)
    input_ids.to(device)
    # dare in pasto al modello

    logits_sentence, logits_token, cls_attention = model(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask)
    pred_score = torch.sigmoid(logits_sentence).item()
    pred = 1 if torch.sigmoid(logits_sentence) >= th else 0

    
    # token cls
    masked_token = input_ids.attention_mask.view(-1) == 1 
    flatten_token_pred = logits_token.squeeze(-1).view(-1)
    flatten_token_pred_masked = torch.masked_select(flatten_token_pred, masked_token)

    token_pred = [float(x >= 0.5) for x in torch.sigmoid(flatten_token_pred_masked)]
    # moltiplico i token pred per i punteggi di attenzione
    attention_scores_masked = torch.masked_select(cls_attention.view(-1), masked_token)
    token_score = (torch.sigmoid(flatten_token_pred_masked) * attention_scores_masked).tolist()
    
        
    pairs = [(tokenizer.decode(id), word,  label, score) for id, word, label, score in zip(\
        torch.masked_select(input_ids.input_ids, masked_token).tolist(), input_ids.words(), token_pred, token_score)]
    pairs_dict = {pair[1]: [] for pair in pairs}
    for pair in pairs:
        pairs_dict[pair[1]].append(pair)
    # rimuovo le key none
    if None in pairs_dict.keys():
        del pairs_dict[None]

    for key in pairs_dict.keys():
        if len(pairs_dict[key]) > 1:
            word = ''
            score = 0
            
            for el in pairs_dict[key]:
                if el[2] == 1:
                    label = 1
                    break
            else:
                label = 0
            
            for el in pairs_dict[key]:
                word += el[0]
                if label == 1:
                    score += el[3]
            pairs_dict[key] = [(word, pairs_dict[key][0][1], label, score)]
    pairs = [(pair[0][0], pair[0][2], pair[0][3]) for pair in pairs_dict.values()]
    result = {'text': text,
               'words': list(filter(lambda x: re.sub(r"[.,!']", "",x), [pair[0] if pair[1] == 1 else '' for pair in pairs])),
              'sentence_pred': pred,
               'pred_score': pred_score,
               'tokens_score': [pair[1] * pair[2] for pair in pairs]
               }
    return result