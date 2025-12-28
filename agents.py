# This document contains the Nature, Alice and Bob LSTM agent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tqdm.auto as tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

class LSTMImageAlice(nn.Module):

    def __init__(self, img_latent_dim, lstm_hidden_dim, vocab_size, max_len, eos, device):
        
        
        self.eos = eos
        self.img_latent_dim = img_latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.vocab_size = vocab_size        
        self.max_len = max_len

        self.embedding = nn.Embedding(self.vocab_size, self.lstm_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.img_latent_dim, self.lstm_hidden_dim),
            nn.ReLU()
        )

        self.lstm_cell =nn.LSTMCell(self.lstm_hidden_dim, self.lstm_hidden_dim) 
        self.output = nn.Linear(self.lstm_hidden_dim, self.vocab_size) 

        # hidden state init
        self.h = None
        self.c = None

        self.device = device 

    def forward(self, target_input):
        # target_input: (Batch, img_latent_dim)
        
        # start with zero-vector embedding
        curr_input = torch.zeros(target_input.size(0), self.lstm_hidden_dim).to(target_input.device)

        log_probs = []
        entropy = []
        messages = [] # List of (Batch) tensors
        probs = []

        # Active mask (1 if sentence continues, 0 if ended)
        active_mask = torch.ones(target_input.size(0)).to(target_input.device)

        for t in range(self.max_len):
            self.h, self.c = self.lstm_cell(curr_input, (self.h, self.c))
            logits = self.output(self.h)
            probs = F.softmax(logits, dim=1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()


            # Save stats
            probs.append(dist.probs() * active_mask)
            log_probs.append(dist.log_prob(action) * active_mask)
            entropy.append(dist.entropy() * active_mask)
            messages.append(action)

            # Update inputs for next step
            curr_input = self.embedding(action)

            # Update mask: if action was EOS (0), mask becomes 0
            is_eos = (action == self.eos).float()
            active_mask = active_mask * (1 - is_eos)

        return torch.stack(messages, dim=1), torch.stack(log_probs, dim=1), torch.stack(entropy, dim=1), torch.stack(probs, dim=1) 

class LSTMImageBob(nn.Module):

    def __init__(self, img_latent_dim, lstm_hidden_dim, vocab_size, eos, device):

        super().__init__()

        
        self.eos = eos
        self.img_latent_dim = img_latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.vocab_size = vocab_size        
        

        self.msgemb = nn.Embedding(self.vocab_size, self.lstm_hidden_dim) 
        self.lstm = nn.LSTM(self.lstm_hidden_dim, self.lstm_hidden_dim, batch_first=True) 
        self.img_encoder = nn.Linear(self.img_latent_dim, self.lstm_hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2*lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim//2, 1) 
        )

        # hidden state init
        self.h = None
        self.c = None

        self.device = device 

    def forward(self, messages, candidates):
        # messages: (Batch, Seq_Len)
        # candidates: (Batch, K, Input_Dim)

        # 1. Encode Message
        # Embed indices
        x = self.msgemb(messages) # (Batch, Seq, Hidden)

        # Run LSTM. We only care about the final hidden state
        # (Assuming the network learns to ignore padding after EOS or EOS is final)
        _, (self.h, _) = self.lstm(x)
        msg_rep = self.h.squeeze(0).unsqueeze(1) # (Batch, 1, Hidden)
        # msg normalization
        msg_reps = msg_rep.expand(-1, candidates.size(1), -1) # (B, K, Hidden) 

        # 2. Encode Images
        img_reps = self.img_encoder(candidates) # (Batch, K, Hidden)

        # 3. Attention / Dot Product
        scores = self.scorer(torch.cat([img_reps, msg_reps], dim=-1)).squeeze(-1)
        dist = torch.distributions.Categorical(logits=scores)
        probs = dist.probs
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, probs

class LSTMImageCentralizedCritic(nn.Module):

    def __init__(self, img_latent_dim, lstm_hidden_dim, vocab_size, eos, device):

        super().__init__()
        
        self.eos = eos
        self.img_latent_dim = img_latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.vocab_size = vocab_size        

        self.fc1 = nn.Linear(self.img_latent_dim, self.lstm_hidden_dim) 
        self.lstm = nn.LSTM(self.lstm_hidden_dim, self.lstm_hidden_dim, batch_first = True) 
        self.fc2 = nn.Linear(self.lstm_hidden_dim, 1) 

        self.device = device 

    def forward(self, candidates, message):
        # candidates: (Batch, K, Input_Dim)
        # message: (Batch, Seq_Len)

        # Flatten candidates
        batch_size = candidates.size(0)
        k = candidates.size(1)
        candidates_flat = candidates.view(batch_size * k, -1) # (Batch*K, Input_Dim)

        # One-hot encode message (take only first symbol for simplicity)
        msg_onehot = F.one_hot(message[:, 0], self.vocab_size).float() # (Batch, Vocab)
        msg_onehot_expanded = msg_onehot.unsqueeze(1).expand(-1, k, -1).contiguous()
        msg_onehot_flat = msg_onehot_expanded.view(batch_size * k, -1) # (Batch*K, Vocab)

        # Concatenate
        critic_input = torch.cat([candidates_flat, msg_onehot_flat], dim=1) # (Batch*K, Input_Dim + Vocab)

        h = F.relu(self.fc1(critic_input))
        h, _ = self.lstm(h.unsqueeze(1)) # (Batch*K, 1, Hidden)
        h = h.squeeze(1) # (Batch*K, Hidden)
        values = self.fc2(h).view(batch_size, k) # (Batch, K)

        # For simplicity, return value corresponding to the first candidate
        return values[:, 0]