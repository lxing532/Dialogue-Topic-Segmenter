import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data_utils import UtteranceDataset
from model_utils import CoherenceNet
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(usage='python train.py -t path/to/dataset -e aws-ai/dse-bert-base -s path/to/checkpoints -m 1')
    parser.add_argument('-t', '--dataset', help='path of dataset for pseudo data generation', default='./data/train/dailydialog')
    parser.add_argument('-r', '--epochs', help='number of training epochs', default=10)
    parser.add_argument('-b', '--batch_size', help='batch size', default=24)
    parser.add_argument('-m', '--margin', help='hyper-parameter of marginal ranking loss', default=1)
    parser.add_argument('-e', '--text_encoder', help='text encoder for utterances', default='aws-ai/dse-bert-base')
    parser.add_argument('-s', '--checkpoints_path', help='path to save checkpoints', default='./checkpoints/')
    args = parser.parse_args()
    return args

def collate_fn(batch):
    return batch

def marginal_ranking_loss(batch, margin):
    # Calculate pairwise differences
    batch_tensor = batch[:, :, 0]

    # Compute individual losses
    # coh1 > coh2, coh1 > coh3, coh2 > coh3
    loss_coh1_coh2 = torch.nn.functional.relu(margin - (batch_tensor[:, 0] - batch_tensor[:, 1]))
    loss_coh1_coh3 = torch.nn.functional.relu(margin - (batch_tensor[:, 0] - batch_tensor[:, 2]))
    loss_coh2_coh3 = torch.nn.functional.relu(margin - (batch_tensor[:, 1] - batch_tensor[:, 2]))

    # Sum and average the losses
    total_loss = (loss_coh1_coh2 + loss_coh1_coh3 + loss_coh2_coh3) / 3
    total_loss = torch.mean(total_loss)

    return total_loss

def validation_metric(sample_list):
    # one triple break down into 3 pairs: a>b, a>c, b>c
    count = 0
    for idx, sample in enumerate(sample_list):
        [a, b, c] = sample
        pairs = [(a, b), (a, c), (b, c)]
        count += sum(1 for x, y in pairs if x > y)

    return count/float(len(sample_list)*3)


def train(model, train_dataloader, val_dataloader, optimizer, epochs, margin, device, checkpoints_path):
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    current_step = 0
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    log_filename = 'training_log.txt' # log file to record progress
    with open(log_filename, 'w') as log_file:
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            log_file.write('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs)); log_file.flush()

            total_loss = 0
            model.train()

            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training...'):

                model.zero_grad()
                output = model(batch)  # prediction

                loss = marginal_ranking_loss(output, margin) # loss computing
                total_loss += loss.item()

                loss.backward()  # back-propagte
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if step % 1000 == 0 and not step == 0: # log every 1000 steps
                    progress_log = f'{step} steps done.... {total_loss / step}\n'
                    print(progress_log, end='')
                    log_file.write(progress_log); log_file.flush()

                    val_res = validation(model, val_dataloader, device)  # apply model from this step to the validation set
                    val_log = f'Validation Results - Step {step}: {val_res}\n'
                    print(val_log)
                    log_file.write(val_log); log_file.flush()

                    # Save model
                    torch.save(model.state_dict(), checkpoints_path+'/cpt_'+str(current_step)+'.pth')  # save the model checkpoint for this step.
                    
                current_step += 1
                
            avg_train_loss = total_loss / len(train_dataloader)  # mean loss for the epoch
            epoch_log = f'=========== The loss for epoch {epoch_i} is: {avg_train_loss}\n'
            print(epoch_log)
            log_file.write(epoch_log); log_file.flush()


def validation(model, val_dataloader, device):
    coherence_scores = []
    model.eval()
    with torch.no_grad():
        for step, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Validating...'):
            output = model(val_batch)
            coh_scores = output[:, :, 0]
            coherence_scores += coh_scores.tolist()
    metric_res = validation_metric(coherence_scores)
    return metric_res
    

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # can delete if not using it.
    # Load settings
    args = parse_args()
    data = args.dataset
    epochs = args.epochs
    batch = args.batch_size
    margin = args.margin
    checkpoints_path = args.checkpoints_path
    
    # cpu or gpu
    use_cuda = torch.cuda.is_available()
    if use_cuda: device = 'cuda'
    else: device = 'cpu'

    # Load encoder model
    model = AutoModel.from_pretrained(args.text_encoder).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    # Create dataset and dataloader
    text_path = data + '/dialogues_text.txt'
    topic_path = data + '/dialogues_topic.txt'
    act_path = data + '/dialogues_act.txt'
    full_dataset = UtteranceDataset(text_path, topic_path, act_path, tokenizer)
    # Determine sizes for train and validation sets
    val_size = int(0.1 * len(full_dataset))  # for 10% validation set
    train_size = len(full_dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Utterance-pair coherence scoring model
    model = CoherenceNet(model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

    # Training loop
    train(model, train_dataloader, val_dataloader, optimizer, epochs, margin, device, checkpoints_path)


if __name__ == "__main__":
    main()
  
