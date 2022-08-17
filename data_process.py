import argparse
import random
from transformers import BertTokenizer
import re

def load_txt(in_fname):
    id2txt = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            id2txt[idx] = [utterance.replace(" __eou__","") for utterance in line.strip().split(" __eou__ ")]
    return id2txt

def load_act(in_fname):
    id2act = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            id2act[idx] = line.strip().split(" ")
    return id2act

def load_topic(in_fname):
    id2topic = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            id2topic[idx] = line.strip()
    return id2topic

text_path = '/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dialogues_text.txt'
topic_path = '/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dialogues_topic.txt'
act_path = '/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dialogues_act.txt'

# load all the dialogues and their features...
txt_dict = load_txt(text_path)
topic_dict = load_topic(topic_path)
act_dict = load_act(act_path)

# extract the utterance pairs with patterns: 2-1, 3-4
tuples = []; win_size = 1; count_no = 0
for idx in range(13118):

    utterances = txt_dict[idx]
    acts = act_dict[idx]
    topic = topic_dict[idx]

    for a_idx in range(len(acts)-1):
        if acts[a_idx] == '2':
            if acts[a_idx+1] == '1':
                positive_sample = [utterances[a_idx], utterances[a_idx+1]]
                utterances_wo_1 = [utterances[i] for i in range(len(utterances)) if acts[i] != '1']
                try:
                    if a_idx-1-win_size < 0:
                        negative_sample_1 = [utterances[a_idx],random.choice(utterances_wo_1[a_idx+1+win_size:])]
                    else:
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_1[:a_idx-win_size]+utterances_wo_1[a_idx+1+win_size:])]
                except:
                    #print('there is no negative sample 1...')
                    count_no += 1
                    negative_sample_1 = []
                sampled_dial = txt_dict[random.choice([key for key, value in topic_dict.items() if value != topic])]
                negative_sample_2 = [utterances[a_idx], random.choice(sampled_dial)]
            
                if negative_sample_1 == []:
                    tmp = [positive_sample, negative_sample_2]
                    tuples.append(tmp)
                else:
                    tmp = [positive_sample, negative_sample_1, negative_sample_2]
                    tuples.append(tmp)

        if acts[a_idx] == '3':
            if acts[a_idx+1] == '4':
                positive_sample = [utterances[a_idx], utterances[a_idx+1]]
                utterances_wo_4 = [utterances[i] for i in range(len(utterances)) if acts[i] != '4']
                try:
                    if a_idx-1-win_size < 0:
                        negative_sample_1 = [utterances[a_idx],random.choice(utterances_wo_4[a_idx+1+win_size:])]
                    else:
                        negative_sample_1 = [utterances[a_idx], random.choice(utterances_wo_4[:a_idx-win_size]+utterances_wo_4[a_idx+1+win_size:])]
                except:
                    #print('there is no negative sample 1...')
                    count_no += 1
                    negative_sample_1 = []
                sampled_dial = txt_dict[random.choice([key for key, value in topic_dict.items() if value != topic])]
                negative_sample_2 = [utterances[a_idx], random.choice(sampled_dial)]
            
                if negative_sample_1 == []:
                    tmp = [positive_sample, negative_sample_2]
                    tuples.append(tmp)
                else:
                    tmp = [positive_sample, negative_sample_1, negative_sample_2]
                    tuples.append(tmp)
    #print(idx)
print(len(tuples))
print(count_no)

sample_num_memory = []

f = open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_pairs.txt',"w+")
for tup in tuples:
    sample_num_memory.append(len(tup))
    for pir in tup:
        sent1 = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', pir[0])
        sent2 = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', pir[1])
        f.write(sent1+'\t\t'+sent2)
        f.write('\n')
f.close()

f = open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_sample_num.txt',"w+")
for sample_size in sample_num_memory:
    f.write(str(sample_size))
    f.write('\n')
f.close()





