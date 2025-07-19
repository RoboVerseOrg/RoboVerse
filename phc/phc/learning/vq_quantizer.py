import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.uniform_(-1.0 / 2, 1.0 / 2)
        self.embedding.weight.data.uniform_(-1.0 / 256, 1.0 / 256)
        # self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim = -1, keepdim=True)  # project to sphere
        # self.embedding.weight.data[:] *= 10
        

    def forward(self, z, return_perplexity=False, return_loss = True):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        if return_loss:
            loss = torch.mean((z_q - z.detach())**2) + self.beta * torch.mean((z_q.detach() - z)**2)
            # loss =  self.beta * torch.mean((z_q.detach() - z)**2)

            # preserve gradients
            z_q = z + (z_q - z).detach()
        else:
            loss = torch.tensor(0.0).to(z.device)
        
        if return_perplexity:
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype) # measuring utilization
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
            return loss, z_q, min_encoding_indices, perplexity
        else:
            return loss, z_q, min_encoding_indices

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super(EmbeddingEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim) 
        
        # weight = weight/weight.norm(dim = -1, keepdim=True) # project to sphere
        
        self.weight = nn.Parameter(weight, requires_grad=False)
        # self.weight.data.uniform_(-1.0 / num_tokens, 1.0 / num_tokens)
        self.weight.data.uniform_(-1.0, 1.0)
        
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False) # counts for how many times the code is used.
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_emb_avg):
        self.update_idxes = new_emb_avg.abs().sum(dim = -1) > 0
        self.embed_avg.data[self.update_idxes] = self.embed_avg.data[self.update_idxes].mul_(self.decay).add(new_emb_avg[self.update_idxes], alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + num_tokens*self.eps) * n)
        embed_normalized = self.embed_avg 
        embed_normalized[self.update_idxes] = self.embed_avg[self.update_idxes] / smoothed_cluster_size.unsqueeze(1)[self.update_idxes]
        self.weight.data.copy_(embed_normalized)
        
        
        

class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5):
        super(EMAVectorQuantizer, self).__init__()

        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z, return_perplexity=False):
        z_flattened = z.view(-1, self.codebook_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        min_encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
        
        if self.training and self.embedding.update:
            encoding_sum = min_encodings.sum(0)
            embed_sum = min_encodings.transpose(0, 1) @ z_flattened
            
            self.embedding.cluster_size_ema_update(encoding_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()
        
        if return_perplexity:
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            return loss, z_q, min_encoding_indices, perplexity
        else:
            return loss, z_q, min_encoding_indices

