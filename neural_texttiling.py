import segeval
from sklearn.metrics import f1_score
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def similarity_computing(texts, tokenizer, text_encoder, mode, device):
    if mode == 'SC':
        ### OPT 1: BI-ENCODER
        #tokenization
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # calculate the sentence embeddings by averaging the embeddings of non-padding words
        with torch.no_grad():
            embeddings = text_encoder(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(device)
            embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)  # mean-pooling
            #embeddings = embeddings[0][:,0,:] # CLS
        scores = torch.cosine_similarity(embeddings[:-1, :], embeddings[1:, :]).cpu().detach().numpy()

    if mode == 'NSP': 
        ### OPT 2: CROSS-ENCODER
        scores = []
        # calculate the NSP probability and use it as score
        with torch.no_grad():
            for i in range(len(texts)-1):
                sent1 = texts[i]
                sent2 = texts[i+1]
                tokenized = tokenizer(sent1, sent2, padding='max_length', max_length = 128, truncation=True, return_tensors='pt')
                logits = text_encoder(**tokenized.to(device)).logits
                probabilities = torch.softmax(logits, dim=1)
                probabilities = probabilities[0][0].item()
                scores.append(probabilities)

    if mode == 'CM':
        ### OPT 3: COHERENCE SCORING
        scores = []
        # calculate the coherence score and use it as score
        with torch.no_grad():
            for i in range(len(texts)-1):
                sent1 = texts[i]
                sent2 = texts[i+1]
                tokenized = tokenizer(sent1, sent2, padding='max_length', max_length = 128, truncation=True, return_tensors='pt')
                output = text_encoder([[tokenized]*3])
                coh_scores = output[:, :, 0]
                scores.append(coh_scores.tolist()[0][0])

    return scores


def depth_computing(scores):

    num_scores = len(scores)
    depth_scores = []

    for i in range(num_scores):
        # Initialize left and right flags with the current score
        left_flag, right_flag = scores[i], scores[i]

        # Search to the left
        for left_index in range(i - 1, -1, -1):
            if scores[left_index] >= left_flag:
                left_flag = scores[left_index]
            else:
                break  # Stop if the score decreases

        # Search to the right
        for right_index in range(i + 1, num_scores):
            if scores[right_index] >= right_flag:
                right_flag = scores[right_index]
            else:
                break  # Stop if the score decreases

        # Calculate the depth score
        depth_score = 0.5 * (left_flag + right_flag - 2 * scores[i])
        depth_scores.append(depth_score)

    return np.array(depth_scores)


def boundaries_to_segments(boundary_indices, total_entries):
    # Initialize the segment sizes list
    segment_sizes = []

    # The first boundary marks the end of the first segment
    previous_boundary = -1
    for boundary in boundary_indices:
        # Add the size of the current segment
        segment_sizes.append(boundary - previous_boundary)
        previous_boundary = boundary

    # Return the segment sizes
    return segment_sizes


def boundaries_to_binary(boundary_indices, total_entries):
    binary_list = [0] * total_entries  # Initialize a list of zeros with the length of total entries
    for index in boundary_indices:
        if 0 <= index < total_entries:  # Check if index is within the range of entries
            binary_list[index] = 1
    binary_list[-1] = 1
    return binary_list


def segments_to_binary(segment_sizes):
    # The total length will be the sum of segment sizes
    total_length = sum(segment_sizes)
    binary_list = [0] * total_length  # Initialize a list of zeros

    # The end index of each segment will be the cumulative sum of the segment sizes
    end_indices = [sum(segment_sizes[:i+1]) for i in range(len(segment_sizes))]
    
    # Place a '1' at each end index, which is the last item of each segment
    for index in end_indices[:-1]:  # Exclude the last index because it's the end of the list
        binary_list[index - 1] = 1  # Subtract 1 because list indices start at 0

    binary_list[-1] = 1
    return binary_list


def TextTiling(dialogue, reference, text_encoder, tokenizer, alpha, mode, device='cpu'):

    # encoding utterances and compute similarity between adjacent utterances
    similarity_scores = similarity_computing(dialogue, tokenizer, text_encoder, mode, device)

    # convert similarity scores --> depth scores
    depth_scores = depth_computing(similarity_scores)

    # threhold computing: threshold = mean - alpha * std
    threshold = depth_scores.mean() + alpha * depth_scores.std()

    # select scores over threshold as boundaries
    boundaries = [i for i in range(len(depth_scores)) if depth_scores[i] > threshold] + [len(dialogue)-1]
    segments = boundaries_to_segments(boundaries, len(dialogue))
    binary_labels = boundaries_to_binary(boundaries, len(dialogue))

    # evaluating
    score_pk = segeval.pk(segments, reference)
    score_wd = segeval.window_diff(segments, reference)
    score_f1 = f1_score(binary_labels, segments_to_binary(reference), labels = [0,1], average='macro')

    return float(score_pk), float(score_wd), float(score_f1), segments

