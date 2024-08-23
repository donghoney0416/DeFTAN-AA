import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Array Geometry Agnostic Speech Enhancement Network

class Network(nn.Module):
    def __init__(self, n_srcs=1, win=512, n_mics=4, n_layers=4, hidden_dim=256, attn_n_head=4, emb_dim=64, emb_ks=4, emb_hs=1, eps=1.0e-5):
        super().__init__()
        self.n_srcs = n_srcs
        self.win = win
        self.hop = win // 2
        self.n_layers = n_layers
        self.n_mics = n_mics
        assert win % 2 == 0

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.in_conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(4, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.att_enc = Att_Encoder()
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(DeFTANBlock(emb_dim, emb_ks, emb_hs, hidden_dim, n_head=attn_n_head, eps=eps))
        self.cross_att = nn.MultiheadAttention(emb_dim, attn_n_head, dropout=0.1, batch_first=True)
        self.LN = nn.LayerNorm(emb_dim)
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.hop)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input: Union[torch.Tensor]) -> Tuple[List[Union[torch.Tensor]], torch.Tensor, OrderedDict]:
        input, rest = self.pad_signal(input)
        B, M, N = input.size()  # batch B, mic M, time samples N

        stft_input = torch.stft(input.view([-1, N]), n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)
        _, F, T, _ = stft_input.size()  # B*M , F= num freqs, T= num frame, 2= real imag
        xi = stft_input.view([B, M, F, T, 2])  # B*M, T, T, 2 -> B, M, T, T, 2
        xi = xi.permute(0, 1, 4, 3, 2).contiguous()  # B, M, 2, T, F
        batch = xi.view([B, M * 2, T, F])  # B, 2*M, T, F
        batch = self.att_enc(batch)        # BM, 4, T, F
        object = self.in_conv(xi.view([B*M, 2, T, F]))      # BM, C, T, F

        batch = self.conv(batch)  # [BM, C, T, F]
        C = batch.shape[1]
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, C, T, F]
            if ii == 0:
                space = self.LN(batch.reshape(B, M, C, T, F).permute(0, 3, 4, 1, 2).reshape(B*T*F, M, C))
                object = object.reshape(B, M, C, T, F).permute(0, 3, 4, 1, 2).reshape(B*T*F, M, C)
                batch = self.cross_att(space, object, object)[0] + space
                batch = batch.reshape(B, T, F, M, C).permute(0, 3, 4, 1, 2).contiguous()
                batch = batch.mean(dim=1)
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([B, 2, T, F]).permute(0, 3, 2, 1).type(input.type())
        istft_input = torch.complex(batch[:, :, :, 0], batch[:, :, :, 1])
        istft_output = torch.istft(istft_input, n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)

        output = istft_output[:, self.hop:-(rest + self.hop)].unsqueeze(1)  # B, 1, T

        return output

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class Att_Encoder(nn.Module):
    def __init__(self, emb_dim=257, heads=4, dim_head=64, dropout=0.1):
        super().__init__()
        self.CT_conv = nn.Conv2d(emb_dim, emb_dim, (3, 3), (1, 1), (1, 1))
        self.C_att = nn.MultiheadAttention(emb_dim, heads, dropout=0.1, batch_first=True)
        self.C_ffw = FeedForward(emb_dim, emb_dim, dropout=0.1)
        self.LN1 = nn.LayerNorm(emb_dim)
        self.LN2 = nn.LayerNorm(emb_dim)

    def forward(self, input):
        B, M, T, F = input.shape
        x = self.CT_conv(input.permute(0, 3, 2, 1).contiguous())    # B, F, T, 2M
        x = x.permute(0, 2, 3, 1).reshape(B*T, -1, F)
        x = self.LN(self.C_att(x, x, x)[0] + x)
        x = self.LN(self.C_ffw(x) + x)
        x = x.reshape(B, T, -1, F).permute(0, 2, 1, 3).contiguous() # B, 2M, T, F

        x = torch.cat((input.unsqueeze(1), x.unsqueeze(1)), dim=1)  # B, 2, 2M, T, F
        x = x.reshape(B, 2, 2, M//2, T, F).reshape(B, 4, M//2, T, F)    # B, 4, M, T, F
        x = x.permute(0, 2, 1, 3, 4).reshape(B*M//2, 4, T, F)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        weight = torch.matmul(F.softmax(k, dim=2).transpose(-1, -2), v) * self.scale
        out = torch.matmul(F.softmax(q, dim=3), weight)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class GSDB(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(out_channels * ((idx > 0) + 1), 2 * out_channels, kernel_size=3, padding=1),
                nn.GLU(dim=1),
            ))

    def forward(self, x):
        B, C, L = x.size()
        g = self.groups
        # x = x.view(B, g, C//g, L).transpose(1, 2).reshape(B, C, L)
        skip = x[:, ::g, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :]], dim=1)
        return output


class DeFTANBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, emb_dim, emb_ks, emb_hs, hidden_channels, n_head=4, eps=1e-5):
        super().__init__()

        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps)
        self.intra_inv = GSDB(in_channels, emb_dim, emb_ks)

        self.intra_att = PreNorm(emb_dim, Attention(emb_dim, n_head, emb_dim, dropout=0.1))
        self.intra_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_channels, dropout=0.1))
        self.intra_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = LayerNormalization4D(emb_dim, eps)
        self.inter_inv = GSDB(in_channels, emb_dim, emb_ks)

        self.inter_att = PreNorm(emb_dim, Attention(emb_dim, n_head, emb_dim, dropout=0.1))
        self.inter_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_channels, dropout=0.1))
        self.inter_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.
        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # F-transformer
        input_ = x
        F_trans = self.intra_norm(input_)  # [B, C, T, Q]
        F_trans = F_trans.transpose(1, 2).contiguous().view(B * T, C, Q)  # [BT, C, Q]
        F_trans = F.unfold(F_trans[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        F_trans = self.intra_inv(F_trans)  # [BT, C, -1]

        F_trans = F_trans.transpose(1, 2)  # [BT, -1, C]
        F_trans = self.intra_att(F_trans) + F_trans
        F_trans = self.intra_ffw(F_trans) + F_trans
        F_trans = F_trans.transpose(1, 2)  # [BT, H, -1]

        F_trans = self.intra_linear(F_trans)  # [BT, C, Q]
        F_trans = F_trans.view([B, T, C, Q])
        F_trans = F_trans.transpose(1, 2).contiguous()  # [B, C, T, Q]
        F_trans = F_trans + input_  # [B, C, T, Q]

        # T-transformer
        input_ = F_trans
        T_trans = self.inter_norm(input_)  # [B, C, T, F]
        T_trans = T_trans.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # [BF, C, T]
        T_trans = F.unfold(T_trans[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        T_trans = self.inter_inv(T_trans)  # [BF, C, -1]

        T_trans = T_trans.transpose(1, 2)  # [BF, -1, C]
        T_trans = self.inter_att(T_trans) + T_trans
        T_trans = self.inter_ffw(T_trans) + T_trans
        T_trans = T_trans.transpose(1, 2)  # [BF, H, -1]

        T_trans = self.inter_linear(T_trans)  # [BF, C, T]
        T_trans = T_trans.view([B, Q, C, T])
        T_trans = T_trans.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        T_trans = T_trans + input_  # [B, C, T, Q]

        return T_trans


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat