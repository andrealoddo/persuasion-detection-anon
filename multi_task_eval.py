import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from torch import cuda
from datasets import load_dataset
import numpy as np
import sys
sys.path.insert(1, '.')
from models import TokenMultiTaskModel
from utils_train import use_multi_task_classification_model

dataset = load_dataset('APauli/Persuasive-Pairs')
path_model = 'repo_anonimus\\model\\'
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(path_model)

model = TokenMultiTaskModel('FacebookAI/xlm-roberta-large', 2, max_length=256)

model.load_state_dict(torch.load(f'{path_model}best_model.pth'))
model.to(device)
model.eval()

# filtraggio somma zero
dataset['train'] = list(filter(lambda x: x['score_1'] + x['score_2'] + x['score_3'] != 0, dataset['train']))

results = []
labels = []
preds = []
for TH in [0.5]:
    results = []
    labels = []
    preds = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset['train']))):
            score_1 = dataset['train'][i]['score_1']
            score_2 = dataset['train'][i]['score_2']
            score_3 = dataset['train'][i]['score_3']

            score_label = score_1 + score_2 + score_3

            text1 = dataset['train'][i]['text1']
            text2 = dataset['train'][i]['text2']

            result1 = use_multi_task_classification_model(text1, model, tokenizer, TH, device)
            result2 = use_multi_task_classification_model(text2, model, tokenizer, TH, device)
            
            token_score1 = sum(result1['tokens_score']) 
            token_score2 = sum(result2['tokens_score'])
            # lambda_ = 0.8
            # score_text1 = (1 - lambda_) * result1['pred_score'] + lambda_ * token_score1
            # score_text2 = (1 - lambda_) * result2['pred_score'] + lambda_ * token_score2
            score_text1 = (result1['pred_score'] + token_score1) / 2 
            score_text2 = (result2['pred_score'] + token_score2) / 2 

            score_pred = max(score_text1, score_text2)

            label = 'text1' if score_label < 0 else 'text2'

            pred = 'unk' if score_text1 == 0 and score_text2 == 0 else 'text1' if score_text2 < score_text1 else 'text2'

            labels.append(label)
            preds.append(pred)
            
            info = {'id': i,
                    'text1': text1,
                    'text1_words': result1['words'],
                    'word_scores1': result1['tokens_score'],
                    'text2': text2,
                    'text2_words': result2['words'],
                    'word_scores2': result2['tokens_score'],
                    'token_score1': token_score1,
                    'token_score2': token_score2,
                    'score_label': score_label,
                    'score_text1': score_text1,
                    'score_text2': score_text2,
                    'score_pred': score_pred, 
                    'label': label,
                    'pred': pred
            }
            results.append(info)

        
    # remove predictions and corrisponding labels with 'unk'
    preds_labels_filtered = [(l, p) for l, p in zip(labels, preds) if p != 'unk']
    if len(preds_labels_filtered) > 0:
        labels, preds = zip(*preds_labels_filtered) # unpacking
    else:
        labels, preds = [], []

    result = {
        'preds': results,
        'unks': len(results) - len(preds_labels_filtered),
        'accuracy': accuracy_score(labels, preds), 
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'f1_score': f1_score(labels, preds, average='macro'),
        'confusion_matrix': confusion_matrix(labels, preds).tolist()
    }

    with open(f'repo_anonimus\\persuasive_pairs_analysis\\persuasive_pairs_multi_task_evaluation.json', 'w') as f:
        json.dump(result, f)