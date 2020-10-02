"""
Teacher that goes from demonstrations -> language
"""

import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class DemonstrationEncoder(nn.Module):
    def __init__(self, n_dirs, n_acts, obs_dim=(7, 7), dir_dim=50, act_dim=100, hidden_dim=512):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n = obs_dim[0]
        m = obs_dim[1]
        self.obs_dim = obs_dim
        self.obs_hidden_dim = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.dir_dim = dir_dim
        self.act_dim = act_dim

        self.n_dirs = n_dirs
        self.n_acts = n_acts

        self.dir_embedding = nn.Embedding(n_dirs, dir_dim, padding_idx=0)
        self.act_embedding = nn.Embedding(n_acts, act_dim, padding_idx=0)

        self.input_dim = self.obs_hidden_dim + dir_dim + act_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, obs, dirs, acts, obs_lens, hidden=None):
        obs_shape = obs.shape
        obs_flat = obs.view(obs_shape[0] * obs_shape[1], *obs_shape[2:])
        obs_flat_enc = self.obs_encoder(obs_flat)
        obs_enc = obs_flat_enc.view(obs_shape[0], obs_shape[1], -1)

        dirs_enc = self.dir_embedding(dirs)
        acts_enc = self.act_embedding(acts)

        inp_enc = torch.cat((obs_enc, dirs_enc, acts_enc), 2)

        packed = pack_padded_sequence(inp_enc, obs_lens, enforce_sorted=False, batch_first=True)

        _, hidden = self.gru(packed, hidden)
        return hidden[-1]


class LanguageDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, enc_out, tgt, tgt_len):
        # embed your sequences
        embed_tgt = self.embedding(tgt)

        # assume 1-layer gru - hidden state expansion
        enc_out = enc_out.unsqueeze(0)

        # No need to decode at the end position, so decrement tgt_len
        tgt_len = tgt_len - 1
        packed_input = pack_padded_sequence(embed_tgt, tgt_len, enforce_sorted=False, batch_first=True)

        # shape = (tgt_len, batch, hidden_dim)
        packed_output, _ = self.gru(packed_input, enc_out)
        logits = self.dense(packed_output.data)

        return logits

    def sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        """Generate from image features using greedy search."""
        with torch.no_grad():
            batch_size = feats.size(0)

            # initialize hidden states using image features
            states = feats.unsqueeze(0)

            # first input is SOS token
            inputs = np.array([sos_index for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in range(20):  # like in jacobs repo
                outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)  # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)  # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)  # inputs: (L=1,B)
                inputs = self.embedding(inputs)  # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * pad_index

            for i in range(batch_size):
                padded_ids[i, : sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths


class Teacher(nn.Module):
    def __init__(self, n_dirs, n_acts, n_lang):
        super().__init__()
        self.encoder = DemonstrationEncoder(n_dirs, n_acts)
        self.decoder = LanguageDecoder(n_lang)

    def forward(self, obs, dirs, acts, obs_lens, langs, lang_lens, hidden=None):
        demos_enc = self.encoder(obs, dirs, acts, obs_lens)

        langs_pred = self.decoder(demos_enc, langs, lang_lens)
        return langs_pred
