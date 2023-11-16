from transformers import AutoModel, AutoTokenizer, AutoModelForNextSentencePrediction
import torch
import argparse
import json
from neural_texttiling import TextTiling
from coherence_model import CoherenceNet
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def data_load(filepath):
    ######
    # This function load dialogue samples and their corresponding ground-truth segments.
    ######
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


def alpha_search(dialogue_data, text_encoder, tokenizer, mode, device, lowerbound, higherbound, step):
    best_alpha = None
    best_pk = float('inf')

    for alpha in tqdm(np.arange(lowerbound, higherbound, step), desc='Searching for best alpha on DEV set'):
        total_pk = 0
        num_samples = len(dialogue_data)
        
        # Evaluate each dialogue at the current alpha value
        for dialogue in dialogue_data:
            pk, _, _, _ = TextTiling(dialogue['utterances'], dialogue['segments'], text_encoder, tokenizer, alpha, mode, device)
            total_pk += pk

        mean_pk = total_pk / num_samples

        if mean_pk < best_pk:
            best_pk = mean_pk
            best_alpha = alpha
    
    # Return the best alpha and its corresponding Pk score
    return best_alpha, best_pk


def parse_args():
    parser = argparse.ArgumentParser(usage='python segment.py -t path/to/data -e text_encoder_name -m CM')
    parser.add_argument('-t', '--dataset', help='path to the dataset', default='dialseg_711.json')
    parser.add_argument('-e', '--text_encoder', help='text encoder for utterances', default='./dse_checkpoints/cpt_1.pth')
    parser.add_argument('-m', '--mode', help='encoder as sequence classification (SC) / next sentence prediction (NSP) / coherence model (CM)', default='CM')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # load settings
    args = parse_args()
    data = args.dataset
    text_encoder_name = args.text_encoder
    mode = args.mode
    
    # cpu or gpu
    use_cuda = torch.cuda.is_available()
    if use_cuda: device = 'cuda'
    else: device = 'cpu'

    # load encoder model
    if mode == 'SC': text_encoder = AutoModel.from_pretrained(text_encoder_name).to(device)     # Sequence Classification
    if mode == 'NSP': text_encoder = AutoModelForNextSentencePrediction.from_pretrained(text_encoder_name).to(device)    # Next Sentence Prediction
    if mode == 'CM': 
        text_encoder = CoherenceNet(AutoModel.from_pretrained('aws-ai/dse-bert-base')) 
        checkpoint = torch.load(text_encoder_name)
        text_encoder.load_state_dict(checkpoint)
        text_encoder.to(device)

    if mode == 'CM': tokenizer = AutoTokenizer.from_pretrained('aws-ai/dse-bert-base')
    else: tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

    # load data (dev data - 1% of all data, test data - 99% of all data)
    dialogue_data = data_load(data)
    dev_data = []; test_data = []
    for dialogue in dialogue_data:
        if dialogue['set'] == 'dev': dev_data.append(dialogue)
        else: test_data.append(dialogue)
        #elif dialogue['set'] == 'test': test_data.append(dialogue)

    # Evaluation starts here ...
    ## Hyper-parameter (alpha) search on dev set
    best_alpha, best_pk = alpha_search(dev_data, text_encoder, tokenizer, mode, device, -2, 2, 0.1)
    print('[INFO] The loaded text encoder is: ', text_encoder_name)
    print('[INFO] The best hyper-parameter (alpha): ', best_alpha)

    ## Evaluation on test set
    total_pk = 0
    total_wd = 0
    total_f1 = 0
    num_samples = len(test_data)
    for i, dialogue in tqdm(enumerate(test_data), total=len(test_data), desc='Evaluating TEST set'):
        pk, wd, f1, pred_segments = TextTiling(dialogue['utterances'], dialogue['segments'], text_encoder, tokenizer, best_alpha, mode, device)
        total_pk += pk
        total_wd += wd
        total_f1 += f1

    # Compute the mean scores
    mean_pk = total_pk / num_samples
    mean_wd = total_wd / num_samples
    mean_f1 = total_f1 / num_samples

    # Print or return the mean scores
    print('-----------------------------------')
    print(f"Mean P_k score: {mean_pk}")
    print(f"Mean WindowDiff score: {mean_wd}")
    print(f"Mean F1 score: {mean_f1}")
    print('-----------------------------------')
        
