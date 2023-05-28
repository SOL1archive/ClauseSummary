from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def generate_seq(model, tokenizer, input):
    generated_ids = model.generate(**input)
    generated_text = tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)
    
    return generated_text

def generate_input_target(model, tokenizer, input, label):
    input_text = tokenizer.decode(input['input_ids'].squeeze(0), skip_special_tokens=True)
    generated_text = generate_seq(model, tokenizer, input)
    target_text = tokenizer.decode(label.squeeze(0), skip_special_tokens=True)
    
    return {
        'input_text': input_text,
        'generated_text': generated_text, 
        'target_text': target_text
    }

def generate_from_data(model, tokenizer, data):
    label = data['labels']
    input_data = dict()
    input_data['input_ids'] = data['input_ids']
    input_data['attention_mask'] = data['attention_mask']

    return generate_input_target(model, tokenizer, input_data, label)

def eval(model, tokenizer, input_seq, label, metric: Callable, options = dict()):
    generated_input_target = generate_input_target(model, tokenizer, input_seq, label)
    score = metric(
        generated_input_target['generated_text'], 
        generated_input_target['target_text'],
        **options
    )

    return score

def eval_from_data(model, tokenizer, dataset, metric: Callable, options = dict()):
    result = []
    for data in dataset:
        label = data['labels']
        input_data = {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
        }

        result.append(eval(model, tokenizer, input_data, label, metric, options))

    return pd.Series(result)

def eval_bleu(model, tokenizer, tokenized_testset):
    bleu_score_lt = []
    for example in tqdm(tokenized_testset):
        output = generate_from_data(model, tokenizer, example)
        try:
            bleu_score = sentence_bleu([output['target_text']], 
                                       output['generated_text'], 
                                       smoothing_function=SmoothingFunction().method1
            )
        except ValueError:
            continue
        bleu_score_lt.append(bleu_score)
    
    return pd.DataFrame({'BLEU': bleu_score_lt})

def eval_rogue(model, tokenizer, tokenized_testset):
    rouge = Rouge()
    rouge_score_dict = dict()
    rouge_score_dict['Precision'] = []
    rouge_score_dict['Recall'] = []
    rouge_score_dict['F1'] = []

    for example in tqdm(tokenized_testset):
        output = generate_from_data(model, tokenizer, example)
        try:
            rouge_score = rouge.get_scores(output['generated_text'], 
                                           output['target_text']
            )
        except ValueError:
            continue
        rouge_score_precision = rouge_score[0]['rouge-2']['p']
        rouge_score_recall = rouge_score[0]['rouge-2']['r']
        rouge_score_f = rouge_score[0]['rouge-2']['f']
        
        rouge_score_dict['Precision'].append(rouge_score_precision)
        rouge_score_dict['Recall'].append(rouge_score_recall)
        rouge_score_dict['F1'].append(rouge_score_f)
    
    return pd.DataFrame(rouge_score_dict)
