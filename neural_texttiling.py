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



### The following functions are for generative language model topic boundary inference (code mostly adapted from: https://github.com/xcfcode/PLM_annotator). ###
from torch.nn import CrossEntropyLoss
import copy

def combine_near_two_utterances(dialog_list):
    """
    Combine two adjacent utterances
    """
    dialogues = []
    for dialog in dialog_list:  # for each dialogue-summary pair
        utterances = dialog  # list: [utterance_1, utterance_2, utterance_3,...,utterance_n]
        utterance_num = len(utterances)
        # combine two adjacent utterances
        dialogue_combine_near_2 = [[utterances[i], utterances[i + 1]] for i in range(utterance_num - 1)]
        assert len(dialogue_combine_near_2) + 1 == len(utterances)
        dialogues.append([utterances, dialogue_combine_near_2])
    return dialogues


def construct_example(context_subwords, response_subwords):
    """
    Construct teacher forcing input-output pair
    example abcd->efg
    we can get: abcd->e; abcde->f; abcdef->g
    """
    examples = []
    context_subwords.append("<|endoftext|>")
    for response_subword in response_subwords:
        input = copy.deepcopy(context_subwords)
        target = response_subword

        if len(input) > 512: input = input[-512:]

        examples.append([input, target])
        context_subwords.append(response_subword)
    return examples


def process_one(dialogue, model, tokenizer, device):
    utterances = dialogue[0]
    dialogue_combine_2 = dialogue[1]

    """step-1 choose the first utterance"""
    first_utterance = utterances[0]
    first_utterance_subwords = tokenizer.tokenize(first_utterance, return_tensors="pt")
    first_utterance_len = len(first_utterance_subwords)

    # init losses list
    processed_subwords = first_utterance_subwords + ["<|endoftext|>"]
    # the first utterance is always important, 100 means high loss.
    losses = [0 for _ in range(first_utterance_len)] + [-100]
    assert len(losses) == len(processed_subwords)

    """step-2 start to process utterance pairs"""
    for pair in dialogue_combine_2:
        context = pair[0]  # assume the first utterance is the context
        response = pair[1]  # assume the second utterance is the response

        context_subwords = tokenizer.tokenize(context, return_tensors="pt")
        response_subwords = tokenizer.tokenize(response, return_tensors="pt")

        # construct teacher forcing examples
        examples = construct_example(context_subwords, response_subwords)

        # extend this response
        processed_subwords.extend(response_subwords)

        for example in examples:
            input = example[0]
            target = example[1]
            input_ids = tokenizer.convert_tokens_to_ids(input)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # [1, seq_len]
            target_ids = tokenizer.convert_tokens_to_ids(target)
            target_ids = torch.tensor(target_ids).unsqueeze(0).to(device)  # [1, 1]

            logits = model(input_ids, return_dict=True).logits[:, -1, :]  # [1, vocab_size]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids)
            losses.append(loss.cpu().tolist())

        # post process
        processed_subwords.append("<|endoftext|>")
        losses.append(-100)

    assert len(processed_subwords) == len(losses)
    return processed_subwords, losses


##### Functions for word loss computing #####
def convert_loss2pseudo_bpe(subwords, losses):
    res = []
    for subword, loss in zip(subwords, losses):
        loss = str(loss)
        loss = loss + "$"
        if "Ġ" in subword:
            loss = "Ġ" + loss
        if "<|endoftext|>" in subword:
            loss = "Ġ" + loss + "Ġ"
        res.append(loss)
    assert len(res) == len(subwords)
    return res


def average_loss(word_level_losses):
    res = []
    losses = word_level_losses.split()
    for loss in losses:
        micro_losses = loss.split("$")[:-1]
        avg_loss = sum([float(micro_loss) for micro_loss in micro_losses]) / len(micro_losses)
        res.append(avg_loss)
    return res


def recover_word_level(subwords, losses):
    #print(subwords, losses)
    losses = convert_loss2pseudo_bpe(subwords, losses)
    dialogue = "".join(subwords).replace("Ġ", " ").replace("<|endoftext|>", " <|endoftext|> ")  # subwords --> dialogue
    word_level_losses = "".join(losses).replace("Ġ", " ")
    #print(dialogue.split(), word_level_losses.split(), len(dialogue.split()), len(word_level_losses.split()))
    assert len(dialogue.split()) == len(word_level_losses.split())
    words = dialogue.split()
    losses = average_loss(word_level_losses)
    assert len(words) == len(losses)
    return words, losses


def process_one_word_loss(subwords, losses):
    words, losses = recover_word_level(subwords, losses)  # recover word-level losses
    return words, losses


def get_loss_for_each_utterance(words, losses):
    assert len(words) == len(losses)

    utterances = []
    utterances_loss = []

    utterance = []
    utterance_loss = []

    for word, loss in zip(words, losses):

        if word == "<|endoftext|>":
            if len(utterance_loss) == 0:
                utterance_loss = [0]
            utterances.append(utterance)
            utterances_loss.append(utterance_loss)
            utterance = []
            utterance_loss = []
        else:
            utterance.append(word)
            utterance_loss.append(loss)
    assert len(utterances) == len(utterances_loss)

    loss_for_each_u = [sum(utterance_loss) / len(utterance_loss) for utterance_loss in utterances_loss]
    return utterances, loss_for_each_u


def TextTiling_glm(dialogue, reference, text_decoder, tokenizer, threshold, device='cpu'):

    dialogs = combine_near_two_utterances([dialogue])

    # utterance loss computing
    subwords, losses = process_one(dialogs[0], text_decoder, tokenizer, device)
    words, losses = process_one_word_loss(subwords, losses)
    utterances, loss_for_each_u = get_loss_for_each_utterance(words, losses)

    # select scores over threshold as boundaries
    boundaries = [i for i in range(len(loss_for_each_u)) if loss_for_each_u[i] > threshold] + [len(dialogue)-1]
    segments = boundaries_to_segments(boundaries, len(dialogue))
    binary_labels = boundaries_to_binary(boundaries, len(dialogue))

    # evaluating
    score_pk = segeval.pk(segments, reference)
    score_wd = segeval.window_diff(segments, reference)
    score_f1 = f1_score(binary_labels, segments_to_binary(reference), labels = [0,1], average='macro')

    return float(score_pk), float(score_wd), float(score_f1), segments
