## This data_processing script is initialized to process the DailyDal dataset as descripted in the SIGDIAL paper.
## You can modify the code to set some settings off for your need.

from torch.utils.data import Dataset
import random
import itertools

###################### FOR DATALOADING ######################
def load_txt(in_fname):
    id2txt = {}
    dup_dict = {}
    dup_list = []
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            if line not in dup_dict.values():
                id2txt[idx] = [utterance.replace(" __eou__","") for utterance in line.strip().split(" __eou__ ")]
                dup_dict[idx] = line
            else:
                dup_list.append(idx)
    return id2txt, dup_list

def load_act(in_fname, dup_list):
    id2act = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            if idx not in dup_list:
                id2act[idx] = line.strip().split(" ")
    return id2act

def load_topic(in_fname, dup_list):
    id2topic = {}
    with open(in_fname) as in_file:
        for idx, line in enumerate(in_file):
            if idx not in dup_list:
                id2topic[idx] = int(line.strip())
    return id2topic

def load_meta(text_path, act_path, topic_path):
    txt_dict, dup_list = load_txt(text_path)
    topic_dict = load_topic(topic_path, dup_list)
    act_dict = load_act(act_path, dup_list)

    return txt_dict, topic_dict, act_dict

def remove_duplicates(txt_dict, topic_dict, act_dict):
    """
    Remove duplicated dialogues from all three dictionaries.
    """
    unique_utterances = {}
    cleaned_topic_dict = {}
    cleaned_act_dict = {}

    for key, utterances in txt_dict.items():
        if utterances not in unique_utterances.values():
            unique_utterances[key] = utterances
            cleaned_topic_dict[key] = topic_dict[key]
            cleaned_act_dict[key] = act_dict[key]

    return unique_utterances, cleaned_topic_dict, cleaned_act_dict


################# FOR POS/NEG SAMPLES SELECTION ################
def pesudo_generation_for_one_sample(utterances, acts, topic, txt_dict, act_dict, topic_dict):
    #print(len(utterances), len(acts))
    sample_triple_for_this_dial = []
    for a_idx in range(len(acts)-1):
        if acts[a_idx] == '2':
            if acts[a_idx+1] == '1':
                anchor = utterances[a_idx]
                postive = utterances[a_idx+1]

                # find the first kind of negative samples (within the same dialogue (same dial act as postive utterance) but not adjacent)
                negtive_minor_list = [utterances[i] for i in range(len(utterances)) if acts[i] != '1' and i != a_idx+1 and i != a_idx-1]

                # find the second kind of negative samples (from dialogue with different topic)
                dial_id = random.choice([key for key, value in topic_dict.items() if value != topic]) # randomly choose one dialogue with different topic
                sampled_utterances = txt_dict[dial_id] 
                sampled_acts = act_dict[dial_id]
                negative_major_list = [sampled_utterances[i] for i in range(len(sampled_utterances)) if sampled_acts[i] != '1']

                for u_n1, u_n2 in itertools.product(negtive_minor_list, negative_major_list):
                    sample_triple_for_this_dial.append((anchor, postive, u_n1, u_n2))

        if acts[a_idx] == '3':
            if acts[a_idx+1] == '4':
                anchor = utterances[a_idx]
                postive = utterances[a_idx+1]

                # find the first kind of negative samples (within the same dialogue (same dial act as postive utterance) but not adjacent)
                negtive_minor_list = [utterances[i] for i in range(len(utterances)) if acts[i] != '4' and i != a_idx+1 and i != a_idx-1]

                # find the second kind of negative samples (from dialogue with different topic)
                dial_id = random.choice([key for key, value in topic_dict.items() if value != topic]) # randomly choose one dialogue with different topic
                sampled_utterances = txt_dict[dial_id] 
                sampled_acts = act_dict[dial_id]
                negative_major_list = [sampled_utterances[i] for i in range(len(sampled_utterances)) if sampled_acts[i] != '4']

                for u_n1, u_n2 in itertools.product(negtive_minor_list, negative_major_list):
                    sample_triple_for_this_dial.append((anchor, postive, u_n1, u_n2))
    
    return sample_triple_for_this_dial
                
def pesudo_generation(txt_dict, act_dict, topic_dict):
    sample_triple = []

    for idx, v in txt_dict.items():
        utterances = txt_dict[idx]
        acts = act_dict[idx]
        topic = topic_dict[idx]
        try:
            sample_triple += pesudo_generation_for_one_sample(utterances, acts, topic, txt_dict, act_dict, topic_dict)
        except:
            print('[Error] Problematic datapoint/dialogue, dropped it...')
            continue

    sample_triple = remove_exact_duplicates(sample_triple)
    return sample_triple

def remove_exact_duplicates(entries):
    """
    Remove entries from the list that have exactly the same elements and return the cleaned list.

    :param entries: List of entries, each entry is a tuple with exactly four elements.
    :return: List of entries after removing duplicates.
    """
    unique_entries = []
    seen = set()
    for entry in entries:
        entry_set = frozenset(entry)  # Convert tuple to a frozenset for immutable set operations
        if entry_set not in seen:
            unique_entries.append(entry)
            seen.add(entry_set)
    
    #print(len(unique_entries), ' unique entries out of ', len(entries))
    return unique_entries


################# MAIN CLASS FOR DATALOADER ####################
class UtteranceDataset(Dataset):
    def __init__(self, text_path, topic_path, act_path, tokenizer, max_length=128):
        self.data = self.__dataBuilder__(text_path, topic_path, act_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor, positive, negative1, negative2 = self.data[idx]

        # construct and encode pos/neg pairs in NSP manner
        pos_pair = self.tokenizer(anchor, positive, padding='max_length', max_length = 128, truncation=True, return_tensors='pt')
        neg1_pair = self.tokenizer(anchor, negative1, padding='max_length', max_length = 128, truncation=True, return_tensors='pt')
        neg2_pair = self.tokenizer(anchor, negative2, padding='max_length', max_length = 128, truncation=True, return_tensors='pt')

        return [pos_pair, neg1_pair, neg2_pair]

    def __dataBuilder__(self, text_path, topic_path, act_path):
        # load all the dialogues and their features...
        txt_dict, topic_dict, act_dict = load_meta(text_path, act_path, topic_path)

        # extract the utterance pairs with patterns: 2-1, 3-4
        training_samples = pesudo_generation(txt_dict, act_dict, topic_dict)
        print('\n[INFO] ', len(training_samples), ' pseduo samples have been finially generated')

        return training_samples
