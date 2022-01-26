import argparse
import random
from transformers import BertTokenizer
import re
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForNextSentencePrediction, AdamW, BertConfig
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn


def MarginRankingLoss(p_scores, n_scores):
    margin = 1
    scores = margin - p_scores + n_scores
    scores = scores.clamp(min=0)

    return scores.mean()

device = 0
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

sample_num_memory = []
id_inputs = []

#for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_sample_num.txt'):
for line in open('/ubc/cs/research/nlp/Linzi/dailydial/dailydial_sample_num.txt'):
    line = line.strip()
    sample_num_memory.append(int(line))

#for line in open('/Users/linzi/Desktop/dialogue_test/training_data/dailydial/dailydial_pairs.txt'):
for line in open('/ubc/cs/research/nlp/Linzi/dailydial/dailydial_pairs.txt'):
    line = line.strip().split('\t\t')
    sent1 = line[0]
    sent2 = line[1]
    encoded_sent1 = tokenizer.encode(sent1, add_special_tokens = True, max_length = 128, return_tensors = 'pt')
    encoded_sent2 = tokenizer.encode(sent2, add_special_tokens = True, max_length = 128, return_tensors = 'pt')
    encoded_pair = encoded_sent1[0].tolist() + encoded_sent2[0].tolist()[1:]
    id_inputs.append(torch.Tensor(encoded_pair))

print('Max sentence length: ', max([len(sen) for sen in id_inputs]))

MAX_LEN = 256
print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

attention_masks = []
for sent in id_inputs:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

# group samples .....
grouped_inputs = []; grouped_masks = []
count = 0
for i in sample_num_memory:
    grouped_inputs.append(id_inputs[count: count+i])
    grouped_masks.append(attention_masks[count: count+i])
    count = count + i
print('The group number is: '+ str(len(grouped_inputs)))
# generate pos/neg pairs ....
print('start generating pos and neg pairs ... ')
pos_neg_pairs = []; pos_neg_masks = []
for i in range(len(grouped_inputs)):
    if len(grouped_inputs[i]) == 2:
        pos_neg_pairs.append(grouped_inputs[i])
        pos_neg_masks.append(grouped_masks[i])
    else:
        pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][1]])
        pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][2]])
        pos_neg_pairs.append([grouped_inputs[i][1], grouped_inputs[i][2]])
        pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][1]])
        pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][2]])
        pos_neg_masks.append([grouped_masks[i][1], grouped_masks[i][2]])

print('there are '+str(len(pos_neg_pairs))+' samples been generated...')
fake_labels = [0]*len(pos_neg_pairs)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(pos_neg_pairs, fake_labels, random_state=2018, test_size=0.8)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(pos_neg_masks, fake_labels, random_state=2018, test_size=0.8)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 12
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

coherence_prediction_decoder = []
coherence_prediction_decoder.append(nn.Linear(768, 768))
coherence_prediction_decoder.append(nn.ReLU())
coherence_prediction_decoder.append(nn.Dropout(p=0.1))
coherence_prediction_decoder.append(nn.Linear(768, 2))
coherence_prediction_decoder = nn.Sequential(*coherence_prediction_decoder)
coherence_prediction_decoder.to(device)

model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = True)
model.cuda(device)
optimizer = AdamW(list(model.parameters())+list(coherence_prediction_decoder.parameters()), lr = 2e-5, eps = 1e-8)

epochs = 10
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


for epoch_i in range(0, epochs):

    total_loss = 0

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0

    model.train()
    coherence_prediction_decoder.train()

    for step, batch in enumerate(train_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        model.zero_grad()
        coherence_prediction_decoder.zero_grad()

        pos_scores = model(b_input_ids[:,0,:], attention_mask=b_input_mask[:,0,:])
        pos_scores = pos_scores[1][-1][:,0,:]
        pos_scores = coherence_prediction_decoder(pos_scores)

        neg_scores = model(b_input_ids[:,1,:], attention_mask=b_input_mask[:,1,:])
        neg_scores = neg_scores[1][-1][:,0,:]
        neg_scores = coherence_prediction_decoder(neg_scores)

        #loss = MarginRankingLoss(pos_scores[0][:,0], neg_scores[0][:,0])
        loss = MarginRankingLoss(pos_scores[:,0], neg_scores[:,0])

        total_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(coherence_prediction_decoder.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print('=========== the loss for epoch '+str(epoch_i)+' is: '+str(avg_train_loss))

    print("")
    print("Running Validation...")

    model.eval()
    coherence_prediction_decoder.eval()

    all_pos_scores = []
    all_neg_scores = []

    for step, batch in enumerate(validation_dataloader):

        if step % 1000 == 0 and not step == 0:
            print(str(step)+' steps done....')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        with torch.no_grad():
            pos_scores = model(b_input_ids[:,0,:], attention_mask=b_input_mask[:,0,:])
            pos_scores = pos_scores[1][-1][:,0,:]
            pos_scores = coherence_prediction_decoder(pos_scores)
            neg_scores = model(b_input_ids[:,1,:], attention_mask=b_input_mask[:,1,:])
            neg_scores = neg_scores[1][-1][:,0,:]
            neg_scores = coherence_prediction_decoder(neg_scores)

        #all_pos_scores += pos_scores[0][:,0].detach().cpu().numpy().tolist()
        #all_neg_scores += neg_scores[0][:,0].detach().cpu().numpy().tolist()
        all_pos_scores += pos_scores[:,0].detach().cpu().numpy().tolist()
        all_neg_scores += neg_scores[:,0].detach().cpu().numpy().tolist()

    labels = []

    for i in range(len(all_pos_scores)):
        if all_pos_scores[i] > all_neg_scores[i]:
            labels.append(1)
        else:
            labels.append(0)

    print(sum(labels)/float(len(all_pos_scores)))

    '''
    PATH = '/scratch/linzi/bert_'+str(epoch_i)
    torch.save(model.state_dict(), PATH)
    '''
    #model.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')
    #tokenizer.save_pretrained('/scratch/linzi/bert_'+str(epoch_i)+'/')


























