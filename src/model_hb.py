import torch.nn as nn
import torch
import math
from diffurec_hb_g import DiffuRec
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
from torch.distributions import Gamma, LogNormal, Weibull

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()

        self.noise_dist = args.noise_dist
        self.g_alpha = args.g_alpha
        self.g_beta = args.g_beta
        self.log_mean = args.log_mean
        self.log_std = args.log_std
        self.w_mean = args.w_mean
        self.w_std = args.w_std
        self.a = args.a

    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t  = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t 

    def reverse(self, item_rep, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t()) # origin
        #scores = torch.matmul(F.normalize(rep_diffu), self.item_embeddings.weight.t())
        scores_pos = scores.gather(1 , labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)
      
        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
       
        return loss



    def loss_diffu_ce(self, rep_diffu, labels):

        #scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t()) ##
        scores = torch.matmul(F.normalize(rep_diffu), self.item_embeddings.weight.t())

        return self.loss_ce(scores, labels.squeeze(-1))


    def diffu_rep_pre(self, rep_diffu): 
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores


    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def knn_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist

    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight ** 2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu ** 2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist


    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))


    def noise_distribution(self, x_start):  
        if self.noise_dist == "gaussian":
            noise = th.randn_like(x_start).to(x_start.device)
            return noise

        elif self.noise_dist == "gaussian_2":
            a = self.a
            noise1 = th.randn_like(x_start) * 0.8
            noise2 = th.randn_like(x_start) * 0.5
            noise = a * noise1 + (1 - a) * noise2
            return noise

        elif self.noise_dist == "gamma":
            g_alpha, g_beta = self.g_alpha, self.g_beta
            n = Gamma(g_alpha, g_beta)
            noise = (n.expand(x_start.shape).sample()).to(x_start.device)  # x_start.shape
            return noise

        elif self.noise_dist == "logNormal":
            mean, std = self.log_mean, self.log_std
            n = LogNormal(mean, std)
            noise = (n.expand(x_start.shape).sample() - 1.0).to(x_start.device)  # x_start.shape
            return noise

        elif self.noise_dist == "gaussian_log":
            mean, std = self.log_mean, self.log_std
            a = self.a
            noise1 = th.randn_like(x_start).to(x_start.device)
            n = LogNormal(mean, std)
            noise2 = (n.expand(x_start.shape).sample() - 1).to(x_start.device)  # x_start.shape
            noise = (a * noise1 + (1 - a) * noise2).to(x_start.device)
            return noise

        elif self.noise_dist == "gaussian_gamma":
            g_alpha, g_beta = self.g_alpha, self.g_beta
            a = self.a
            noise1 = th.randn_like(x_start).to(x_start.device)
            n = Gamma(g_alpha, g_beta)
            noise2 = (n.expand(x_start.shape).sample()).to(x_start.device)  # x_start.shape
            noise = a * noise1 + (1 - a) * noise2
            return noise

        elif self.noise_dist == "weibull":
            mean, std = self.w_mean, self.w_std
            n = Weibull(mean, std)
            noise = (n.expand(x_start.shape).sample() - 1.0).to(x_start.device)  # x_start.shape
            return noise

        elif self.noise_dist == "wei2":
            mean, std = self.w_mean, self.w_std
            a = self.a
            n1 = Weibull(mean, std)
            noise1 = n1.expand(x_start.shape).sample()  # x_start.shape
            n2 = Weibull(mean + 0.5, std)
            noise2 = n2.expand(x_start.shape).sample()
            noise = a * noise1 + (1 - a) * noise2
            return noise

        elif self.noise_dist == "gaussian_wei":
            mean, std = self.w_mean, self.w_std
            a = self.a
            noise1 = th.randn_like(x_start).to(x_start.device)
            n = Weibull(mean, std)
            noise2 = (n.expand(x_start.shape).sample() - 1.0).to(x_start.device)  # x_start.shape
            noise = a * noise1 + (1 - a) * noise2
            return noise

        elif self.noise_dist == "bernoulli":
            noise = (th.bernoulli(th.ones_like(x_start) * 0.5) * 2 - 1.)
            return noise

    def forward(self, sequence, tag, train_flag=True):
        seq_length = sequence.size(1)

        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        item_embeddings = self.LayerNorm(item_embeddings)

        mask_seq = (sequence>0).float()

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H
            rep_diffu, rep_item, weights, t = self.diffu_pre(item_embeddings, tag_emb, mask_seq)

            item_rep_dis = None
            seq_rep_dis = None
        else:
            #noise_x_t = th.randn_like(item_embeddings[:,-1,:]) ##
            noise_x_t = self.noise_distribution(noise_x_t) #


            rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq) ##
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre
