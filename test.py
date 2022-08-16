import os
import numpy as np
from numpy import random as np_random
#import random
import copy
import itertools
from os import listdir
from os.path import isfile, join
import shutil
import segeval
import re
from transformers import BertTokenizer
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForNextSentencePrediction
import statistics
from sklearn.metrics import mean_absolute_error, f1_score

def depth_score_cal(scores):
	output_scores = []
	for i in range(len(scores)):
		lflag = scores[i]; rflag = scores[i];
		if i == 0:
			hl = scores[i]
			for r in range(i+1,len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
		elif i == len(scores):
			hr = scores[i]
			for l in range(i-1, -1, -1):
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		else:
			for r in range(i+1,len(scores)):
				if rflag <= scores[r]:
					rflag = scores[r]
				else:
					break
			for l in range(i-1, -1, -1):
				if lflag <= scores[l]:
					lflag = scores[l]
				else:
					break
		depth_score = 0.5*(lflag+rflag-2*scores[i])
		output_scores.append(depth_score)

	return output_scores



device = 0
MODEL_PATH = '/scratch/linzi/bert_9'
#MODEL_PATH = 'bert-base-uncased'
model = BertForNextSentencePrediction.from_pretrained(MODEL_PATH, num_labels = 2, output_attentions = False, output_hidden_states = False)
model.cuda(device)
'''
MODEL_PATH = '/scratch/linzi/bert_9'
model.load_state_dict(torch.load(MODEL_PATH ,map_location=device))
'''
model.eval()

path_input_docs = '/ubc/cs/research/nlp/Linzi/dailydial/doc2dial_data/'
input_files = [f for f in listdir(path_input_docs) if isfile(join(path_input_docs, f))]
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)

c = 0
pick_num = 3
score_wd = 0; score_mae = 0; score_f1 = 0; score_pk = 0;
dp_var = []

for file in input_files:

	if file not in ['.DS_Store', '196']:
	#if file not in ['.DS_Store']:
		print('*********** The current file is : '+ file + '***********')
		text = []
		id_inputs = []
		depth_scores = []
		seg_r_labels = []; seg_r = [];
		tmp = 0
		for line in open('/ubc/cs/research/nlp/Linzi/dailydial/doc2dial_data/'+file):
			if '================' not in line.strip():
				text.append(line.strip())
				seg_r_labels.append(0)
				tmp += 1
			else:
				seg_r_labels[-1] = 1
				seg_r.append(tmp)
				tmp = 0
				
		seg_r.append(tmp)

		for i in range(len(text)-1):
			sent1 = text[i]
			sent2 = text[i+1]
			encoded_sent1 = tokenizer.encode(sent1, add_special_tokens = True, max_length = 128, return_tensors = 'pt')
			encoded_sent2 = tokenizer.encode(sent2, add_special_tokens = True, max_length = 128, return_tensors = 'pt')
			encoded_pair = encoded_sent1[0].tolist() + encoded_sent2[0].tolist()[1:]
			id_inputs.append(torch.Tensor(encoded_pair))

		MAX_LEN = 256
		id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
		attention_masks = []
		for sent in id_inputs:
			att_mask = [int(token_id > 0) for token_id in sent]
			attention_masks.append(att_mask)

		test_inputs = torch.tensor(id_inputs).to(device)
		test_masks = torch.tensor(attention_masks).to(device)

		scores = model(test_inputs, attention_mask=test_masks)
		scores = torch.sigmoid(scores[0][:,0]).detach().cpu().numpy().tolist()

		depth_scores = depth_score_cal(scores)
		#print(depth_scores)

		#boundary_indice = np.argsort(np.array(depth_scores))[-pick_num:]
	
		threshold = sum(depth_scores)/(len(depth_scores))-0.1*statistics.stdev(depth_scores)
		dp_var.append(statistics.stdev(depth_scores))
		boundary_indice = []
	
		seg_p_labels = [0]*(len(depth_scores)+1)
		
		for i in range(len(depth_scores)):
			if depth_scores[i] > threshold:
				boundary_indice.append(i)
		
		for i in boundary_indice:
			seg_p_labels[i] = 1

		tmp = 0; seg_p = []
		for fake in seg_p_labels:
			if fake == 1:
				tmp += 1
				seg_p.append(tmp)
				tmp = 0
			else:
				tmp += 1
		seg_p.append(tmp)

		#print(depth_scores)
		#print(threshold)
		#print(seg_p)
		#print(seg_r)

		score_wd += segeval.window_diff(seg_p, seg_r)
		score_pk += segeval.pk(seg_p, seg_r)
		score_mae += sum(list(map(abs, np.array(seg_r_labels)-np.array(seg_p_labels))))
		score_f1 += f1_score(seg_r_labels, seg_p_labels, labels = [0,1], average='macro')
		print(c)
		print(seg_r_labels)
		print(seg_p_labels)
		c += 1

print(c)
print('pk: ', score_pk/c)
print('wd: ', score_wd/c)
print('mae: ', score_mae/c)
print('f1: ', score_f1/c)
print('dp variance: ', sum(dp_var)/c)



