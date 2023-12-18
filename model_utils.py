import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch.nn as nn


class CoherenceNet(torch.nn.Module):
    def __init__(self, bert_model, device):
        super(CoherenceNet, self).__init__()
        self.bert = bert_model
        self.coherence_decoder = nn.Sequential(
                                    nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(768, 2)
                                )
        self.device = device

    def forward(self, batch):

        output = []

        for idx, sample in enumerate(batch):
            pos_output = self.bert(**sample[0].to(self.device))
            neg1_output = self.bert(**sample[1].to(self.device))
            neg2_output = self.bert(**sample[2].to(self.device))

            pos_output = pos_output.last_hidden_state[:, 0, :]
            neg1_output = neg1_output.last_hidden_state[:, 0, :]
            neg2_output = neg2_output.last_hidden_state[:, 0, :]

            coherence_output_pos = self.coherence_decoder(pos_output)
            coherence_output_neg1 = self.coherence_decoder(neg1_output)
            coherence_output_neg2 = self.coherence_decoder(neg2_output)

            output.append(torch.stack([F.softmax(coherence_output_pos.squeeze(0)), 
                                       F.softmax(coherence_output_neg1.squeeze(0)), 
                                       F.softmax(coherence_output_neg2.squeeze(0))], dim=0))

        return torch.stack(output, dim=0)
        
