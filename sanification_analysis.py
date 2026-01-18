import json
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm
import re
from torch import cuda
import logging
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from model import use_multi_task_classification_model
from model import TokenMultiTaskModel


model_sizes = ['0.6B', '1.7B', '4B', '8B']
model_size = model_sizes[2]


with open(f'repo_anonimus\\rephrase\\purification_results_Qwen3-{model_size}.json', 'r') as f:
    pure = json.load(f)

pure_texts = pure['pure']
orig_texts = pure['origin']

path_model = 'repo_anonimus\\model\\'
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(path_model)

model = TokenMultiTaskModel('FacebookAI/xlm-roberta-large', 2, max_length=256)

model.load_state_dict(torch.load(f'{path_model}best_model.pth'))
model.to(device)
model.eval()
TH = 0.5
score_pure = []
origin_score = []
r1_recall = []
r1_precision = []
r1_f1 = []
rl_recall = []
rl_precision = []
rl_f1 = []

cos_sims = []

comb_score = []

J = []
words_analisis = {
    'origin_text': [],
    'pure_text': [],
    'origin_words': [],
    'pure_words': [],
    'origin_score': [],
    'pure_score': []
}
model_encorder = SentenceTransformer('sentence-transformers/LaBSE')
for text, text_old  in tqdm(zip(pure_texts, orig_texts)):
    text = text.replace('<|im_end|>', '')
    text = text.replace('<answer>', '')
    text = text.replace('</answer>', '')
    
    with torch.no_grad():
        result_new = use_multi_task_classification_model(text, model, tokenizer, TH, device)
        result_old = use_multi_task_classification_model(text_old, model, tokenizer, TH, device)
    
    token_score_new = sum(result_new['tokens_score']) 
    token_score_old = sum(result_old['tokens_score'])
    
    score_text_new = (result_new['pred_score'] + token_score_new) / 2 
    score_text_old = (result_old['pred_score'] + token_score_old) / 2 
    
    words_analisis['origin_text'].append(text_old)
    words_analisis['pure_text'].append(text)
    words_analisis['origin_words'].append(result_old['words'])
    words_analisis['pure_words'].append(result_new['words'])
    words_analisis['origin_score'].append(score_text_old)
    words_analisis['pure_score'].append(score_text_new)

    score_pure.append(score_text_new)
    origin_score.append(score_text_old)

    score_text = np.array(score_text_new)
    old_score = np.array(score_text_old)
   
    compared_scores1 = (score_text <= old_score).astype(float)
    
    combined_scores = ((score_text + compared_scores1) / 2)
    
    comb_score.append(combined_scores)

    emb1 = model_encorder.encode(text, convert_to_tensor=True)
    emb2 = model_encorder.encode(text_old, convert_to_tensor=True)
    
    cos_sim = util.cos_sim(emb1, emb2).item()

    cos_sims.append(cos_sim)

    joined_measure = np.array(cos_sim) * np.array(combined_scores)
    J.append(joined_measure)


    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(text, text_old)
    
    r1_precision.append(scores['rouge1'].precision)
    r1_recall.append(scores['rouge1'].recall)
    r1_f1.append(scores['rouge1'].fmeasure)

    rl_precision.append(scores['rougeL'].precision)
    rl_recall.append(scores['rougeL'].recall)
    rl_f1.append(scores['rougeL'].fmeasure)


with open(f'repo_anonimus\\sanification_results\\purification_results_score_{model_size}.json', 'w') as f:
    json.dump({'orig_score': origin_score,
                'pure_score': score_pure,

                'r1_precision': np.mean(r1_precision),
                'r1_recall': np.mean(r1_recall),
                'r1_f1': np.mean(r1_f1),

                'rl_precision': np.mean(rl_precision),
                'rl_recall': np.mean(rl_recall),
                'rl_f1': np.mean(rl_f1), 
                'cosine_similarity': np.mean(cos_sims),
                'STA': np.mean(comb_score),
                'joined_measure': np.mean(J),}, f)
with open(f'repo_anonimus\\sanification_results\\purification_results_words_analysis_{model_size}.json', 'w') as f:
    json.dump(words_analisis, f)

num_el_pured = len([el for el in zip(origin_score, score_pure) if el[0] > el[1]])
num_el_not_pured = len([el for el in zip(origin_score, score_pure) if el[0] <= el[1]])
print(f'Number of elements that were purified: {num_el_pured}')
print(f'Number of elements that were not purified: {num_el_not_pured}')
print(f'Percentage of elements that were purified: {num_el_pured / (num_el_pured + num_el_not_pured) * 100:.2f}%') 