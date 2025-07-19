import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1],
                                            device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):

    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_frames,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 ablation=None,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim,
                                                       self.dropout)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        x = self.sequence_pos_encoder(x)

        # transformer layers
        final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        # get the average of the output
        z = final.mean(axis=0)

        # extract mu and logvar
        mu = self.mu_layer(z)
        logvar = self.sigma_layer(z)
     

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Module):

    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_frames,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 ablation=None,
                 **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(
                self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer, num_layers=self.num_layers)

        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch[
            "lengths"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                # shift the latent noise vector to be the action noise
                if self.ablation != "average_encoder":
                    z = z + self.actionBiases[y]

                z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        batch["output"] = output
        return batch

def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """
    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x

class CausalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        # self.attn_drop = nn.Dropout(0.1)
        # self.resid_drop = nn.Dropout(0.1)

    def forward(self, x, mask=None, tgt_mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x).reshape(b, n, h, -1).transpose(1, 2)
        kv = self.to_kv(x).reshape(b, n, 2, h, -1).transpose(2, 3)

        k, v = kv[..., 0, :], kv[...,1, :]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask[None, None, :, :].float()
            dots.masked_fill_(mask==0, float('-inf'))

        # if tgt_mask is not None:
        #     tgt_mask = tgt_mask[:, None, :, :].float()
        #     tgt_mask = tgt_mask.transpose(2, 3) * tgt_mask
        #     dots.masked_fill_(tgt_mask==1, float('-inf'))


        attn = dots.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        #out =  self.resid_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, heads=heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, tgt_mask):
        b, s, h = x.shape
        mask = torch.tril(torch.ones(s, s)).bool().to(x.device)
        x = x + self.attn(self.norm1(x), mask=mask, tgt_mask=tgt_mask)
        x = x + self.ff(self.norm2(x))
        return x