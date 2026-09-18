import math
import os
import json
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import time
import gc
import random
import logging
import warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import weight_norm

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def dataset_size_tag(dataset_size):
    return f"num_{dataset_size}"


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 🌟 极速版：挂载 HDF5 超级文件的自定义数据集类
# ==========================================
class FastHDF5Dataset(Dataset):
    def __init__(self, h5_path):
        print(f"🚀 正在挂载 HDF5 高速数据集: {h5_path} ...")
        self.h5_path = h5_path
        self.file = None # 延迟到 __getitem__ 中打开，防止多进程 (DataLoader workers) 冲突
        
        # 预先获取数据集长度
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"未能找到 HDF5 文件: {self.h5_path}")
            
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['labels'].shape[0]
            
        self.num_classes = 14
        print(f"📦 挂载完毕！共检测到 {self.length} 条样本。")

    def __len__(self):
        return self.length


# 🌟 终极防爆魔法：在跨进程传递前，主动丢弃打开的 C 语言文件指针
    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        if 'x_data' in state:
            state['x_data'] = None
        if 'y_data' in state:
            state['y_data'] = None
        return state

    def __getitem__(self, idx):
        # 🌟 核心：确保在子进程中安全打开文件
        if self.file is None:
            #self.file = h5py.File(self.h5_path, 'r', swmr=True) # 启用 SWMR (Single Writer Multiple Reader) 模式极速读取
            self.file = h5py.File(self.h5_path, 'r')
            self.x_data = self.file['data']
            self.y_data = self.file['labels']
            
        # 连续读取，速度是碎文件的几十倍
        X_np = self.x_data[idx]
        Y_np = self.y_data[idx]
        
        # 转回 PyTorch 张量
        X_tensor = torch.from_numpy(X_np).clone()
        #X_tensor = torch.tensor(X_np, dtype=torch.uint8)  
        Y_tensor = torch.tensor(Y_np, dtype=torch.long)
        
        return X_tensor, Y_tensor


# ==========================================
# 基础特征提取编码器 (保持不变)
# ==========================================
class CircularPPIEncoder(nn.Module):
    def __init__(self, out_features=1):
        super(CircularPPIEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=out_features, kernel_size=5, padding=2, padding_mode='circular')
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, ppi):
        x = ppi.squeeze(1)  
        return self.relu(self.bn(self.conv(x))) 

class LinearSTFTEncoder(nn.Module):
    def __init__(self, out_features=1):
        super(LinearSTFTEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=out_features, kernel_size=5, padding=2, padding_mode='zeros')
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, stft):
        x = stft.squeeze(1).transpose(1, 2) 
        return self.relu(self.bn(self.conv(x)))

# 🌟 3. 原有的 STFT 图像编码器 (保持不变，使用零填充)
class RadarImageEncoder(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),  
            
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)) 
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

# ==========================================
# 基础时序组件 (保持不变)
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V 线性映射层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 🌟 新增：LayerNorm 层。与残差机制搭配，确保数值稳定
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x 的形状: (Batch_size, Time_steps, Embed_dim)
        B, T, E = x.shape

        # 🌟 关键修改 1：保存一份原始输入特征，作为我们的“残差主干”
        residual = x 

        # 🌟 新增：采用最稳健的 Pre-Norm 范式，先进行层归一化
        x_norm = self.layer_norm(x) 

        # 计算 Q, K, V (基于归一化后的特征)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)

        # 掩码机制（如果传入了 mask）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化获取注意力权重，并加上 Dropout
        attn = self.dropout(torch.softmax(scores, dim=-1))

        # 注意力权重乘以 V
        out = torch.matmul(attn, v)  # (B, num_heads, T, head_dim)

        # 恢复张量形状为 (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # 线性映射还原特征
        out = self.out_proj(out)
        
        # 🌟 关键修改 2：特征融合机制！原始光流特征 (residual) + 注意力突变特征 (out)
        return residual + out

class MultiHeadSelfAttention_noresidual(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention_noresidual, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, T, E = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, T, E)
        return self.out_proj(out)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding_size = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
# ==========================================
# 🌟 创新组件 1：PyTorch 原生 ConvLSTM 单元
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # 核心：用 Conv2d 替换了传统 LSTM 内部的 Linear 矩阵乘法
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # 在通道维度拼接输入特征图和上一时刻的隐藏特征图
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SymmetricDownsampleEncoderx32(nn.Module):
    def __init__(self, in_channels=2, out_channels=32):
        super().__init__()
        # 接收 2 通道 (PPI + STFT)，进行联合二维空间降维以保护显存
        self.net = nn.Sequential(
            # ==========================================
            # 第一阶段：256 -> 128 -> 64 (整体下采样 4 倍)
            # ==========================================
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2), # 尺寸: 256 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 尺寸: 128 -> 64
            
            # ==========================================
            # 第二阶段：64 -> 64 -> 32 (整体下采样 2 倍)
            # ==========================================
            # 🌟 核心修改点：将 stride=2 改为 stride=1，保留分辨率
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1), # 尺寸: 64 -> 64
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 尺寸: 64 -> 32
        )
        
    def forward(self, x):
        return self.net(x)


class SymmetricDownsampleEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=32):
        super().__init__()
        # 接收 2 通道 (PPI + STFT)，进行联合二维空间降维以保护显存
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 -> 64
            
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64 -> 16
        )
        
    def forward(self, x):
        return self.net(x)


    

class Baseline_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.flat_dim = 2 * 256 * 256
        self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
        self.rnn = nn.RNN(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat = x.view(B, T, -1)
        
        # 🌟 救命级防爆墙：强制关闭 AMP，使用 float32 运算 13万维巨型矩阵
        with torch.cuda.amp.autocast(enabled=False):
            embed_out = F.relu(self.linear_embed(x_flat.float()))
            rnn_out, _ = self.rnn(embed_out)
            
        return self.fc(self.dropout(rnn_out.mean(dim=1)))

# class Baseline_BIGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
#         super().__init__()
#         self.flat_dim = 2 * 256 * 256
#         self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
#         self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
#                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         B, T, C, H, W = x.size()
#         x_flat = x.view(B, T, -1)
#         embed_out = F.relu(self.linear_embed(x_flat))
        
#         gru_out, _ = self.gru(embed_out)
#         return self.fc(self.dropout(gru_out.mean(dim=1)))

# class Baseline_BILSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
#         super().__init__()
#         self.flat_dim = 2 * 256 * 256
#         self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
#         self.lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
#                             batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         B, T, C, H, W = x.size()
#         x_flat = x.view(B, T, -1)
#         embed_out = F.relu(self.linear_embed(x_flat))
        
#         lstm_out, _ = self.lstm(embed_out)
#         return self.fc(self.dropout(lstm_out.mean(dim=1)))


class Baseline_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.flat_dim = 2 * 256 * 256
        self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
        # 修改1：bidirectional=False (单向GRU)
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        
        # 修改2：因为是单向，fc层输入维度从 hidden_size * 2 改为 hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat = x.view(B, T, -1)
        embed_out = F.relu(self.linear_embed(x_flat))
        
        gru_out, _ = self.gru(embed_out)
        return self.fc(self.dropout(gru_out.mean(dim=1)))

class Baseline_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.flat_dim = 2 * 256 * 256
        self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
        # 修改1：bidirectional=False (单向LSTM)
        self.lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        
        # 修改2：因为是单向，fc层输入维度从 hidden_size * 2 改为 hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat = x.view(B, T, -1)
        embed_out = F.relu(self.linear_embed(x_flat))
        
        lstm_out, _ = self.lstm(embed_out)
        return self.fc(self.dropout(lstm_out.mean(dim=1)))



class Baseline_BiGRU_SA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64, attn_heads=2):
        super().__init__()
        # 直接将 2通道 256x256 的图像展平为 131072 维的超长向量
        self.flat_dim = 2 * 256 * 256
        self.linear_embed = nn.Linear(self.flat_dim, hidden_size * 2)
        
        # 双向 GRU
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        
        # 时序自注意力层 (双向 GRU 的输出维度是 hidden_size * 2)
        self.embed_dim = hidden_size * 2
        self.self_attn = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=attn_heads, dropout=dropout)
        
        self.fc = nn.Linear(self.embed_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        
        # 物理降维打击：直接把完整的空间二维流形拉成一根长面条
        x_flat = x.view(B, T, -1)
        embed_out = F.relu(self.linear_embed(x_flat))
        
        # GRU 时序建模
        gru_out, _ = self.gru(embed_out)
        
        # 自注意力机制计算关键帧权重
        attn_out = self.self_attn(gru_out, mask=mask)
        
        # 对时间轴进行均值池化后分类输出
        return self.fc(self.dropout(attn_out.mean(dim=1)))


class Baseline_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.flat_dim = 2 * 256 * 256
        embed_dim = hidden_size * 2
        
        self.linear_embed = nn.Linear(self.flat_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, 
                                                    dim_feedforward=hidden_size * 4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat = x.view(B, T, -1)
        embed_out = F.relu(self.linear_embed(x_flat))
        
        cnn_out = self.pos_encoder(embed_out)
        trans_out = self.transformer(cnn_out)
        return self.fc(self.dropout(trans_out.mean(dim=1)))



# 2. 基线模型：纯 GRU
class Baseline_CNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.encoder_ppi = RadarImageEncoder(cnn_channels)
        self.encoder_stft = RadarImageEncoder(cnn_channels)
        self.gru = nn.GRU(input_size=cnn_channels * 2, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        ppi = x[:, :, 0:1, :, :].contiguous().view(B * T, 1, H, W)
        stft = x[:, :, 1:2, :, :].contiguous().view(B * T, 1, H, W)
        feat_ppi = self.encoder_ppi(ppi)     
        feat_stft = self.encoder_stft(stft)  
        cnn_out = torch.cat([feat_ppi, feat_stft], dim=1).view(B, T, -1)
        
        gru_out, _ = self.gru(cnn_out)
        return self.fc(self.dropout(gru_out.mean(dim=1)))

class Aligned_Baseline_CNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super().__init__()
        
        # 1. 严格统一前端：使用与主推模型完全相同的联合降维编码器 (处理双通道 PPI+STFT)
        self.shared_encoder = SymmetricDownsampleEncoder(in_channels=2, out_channels=cnn_channels)
        
        # 2. 空间特征展平：将二维空间特征图强行压缩为一维向量
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. 标准 GRU (注意：此时 input_size 已变为 cnn_channels，不再是 cnn_channels * 2)
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        
        # 4. 分类器 (双向 GRU，因此 hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W)
        B, T, C, H, W = x.size()
        
        # 将 Batch 和 Time 维度合并，送入 2D 共享编码器
        x_flat = x.view(B * T, C, H, W)
        
        # 提取 2D 空间特征: (B*T, cnn_channels, H_out, W_out)
        feat_2d = self.shared_encoder(x_flat)
        
        # 物理降维打击：使用全局池化彻底抹除空间二维光流特征
        # (B*T, cnn_channels, H_out, W_out) -> (B*T, cnn_channels, 1, 1) -> (B, T, cnn_channels)
        feat_1d = self.gap(feat_2d).view(B, T, -1)
        
        # 时序建模
        gru_out, _ = self.gru(feat_1d)
        
        # 对时间轴进行均值池化，然后分类输出
        return self.fc(self.dropout(gru_out.mean(dim=1)))


# 3. 基线模型：纯 LSTM
class Baseline_CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.encoder_ppi = RadarImageEncoder(cnn_channels)
        self.encoder_stft = RadarImageEncoder(cnn_channels)
        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        ppi = x[:, :, 0:1, :, :].contiguous().view(B * T, 1, H, W)
        stft = x[:, :, 1:2, :, :].contiguous().view(B * T, 1, H, W)
        feat_ppi = self.encoder_ppi(ppi)     
        feat_stft = self.encoder_stft(stft)  
        cnn_out = torch.cat([feat_ppi, feat_stft], dim=1).view(B, T, -1)
        
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(self.dropout(lstm_out.mean(dim=1)))



class Aligned_Baseline_CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super().__init__()
        
        # 1. 严格统一：使用与主推模型完全相同的联合降维编码器
        self.shared_encoder = SymmetricDownsampleEncoder(in_channels=2, out_channels=cnn_channels)
        
        # 2. 空间特征展平：将 2D 特征图压缩为 1D 向量，供标准 LSTM 使用
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. 标准 LSTM (输入维度为 cnn_channels)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
                            
        # 4. 分类器 (双向 LSTM 输出维度翻倍)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W)
        B, T, C, H, W = x.size()
        
        # 合并 B 和 T 进行 2D 卷积
        x_flat = x.view(B * T, C, H, W)
        
        # 提取空间特征: (B*T, cnn_channels, H_out, W_out)
        feat_2d = self.shared_encoder(x_flat)
        
        # 破坏空间拓扑：用全局池化将其展平为 1D，模拟传统方法的空间信息丢失
        # (B*T, cnn_channels, 1, 1) -> (B, T, cnn_channels)
        feat_1d = self.gap(feat_2d).view(B, T, -1)
        
        # 时序建模
        lstm_out, _ = self.lstm(feat_1d)
        
        # 取时间维度的平均值进行分类
        return self.fc(self.dropout(lstm_out.mean(dim=1)))


class Baseline_CNN_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=64):
        super().__init__()
        self.encoder_ppi = RadarImageEncoder(cnn_channels)
        self.encoder_stft = RadarImageEncoder(cnn_channels)
        embed_dim = cnn_channels * 2
        
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, 
                                                    dim_feedforward=hidden_size * 4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        ppi = x[:, :, 0:1, :, :].contiguous().view(B * T, 1, H, W)
        stft = x[:, :, 1:2, :, :].contiguous().view(B * T, 1, H, W)
        feat_ppi = self.encoder_ppi(ppi)     
        feat_stft = self.encoder_stft(stft)  
        cnn_out = torch.cat([feat_ppi, feat_stft], dim=1).view(B, T, -1)
        
        # Transformer 需要位置编码注入时序先验
        cnn_out = self.pos_encoder(cnn_out)
        trans_out = self.transformer(cnn_out)
        return self.fc(self.dropout(trans_out.mean(dim=1)))



# ==========================================
# 🌟 2023 ICLR 顶会轻量化视频 SOTA: UniFormerV2 (极速定制版)
# 彻底解决 VideoSwin 的算力灾难，提供极低显存的高效时空融合
# ==========================================

class LocalUniBlock(nn.Module):
    """浅层局部时空提取：基于 3D 深度可分离卷积，极速无显存压力"""
    def __init__(self, dim):
        super().__init__()
        # 3D Depthwise 卷积 (提取局部时空光流)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.BatchNorm3d(dim)
        # Pointwise 卷积扩展特征
        self.pwconv1 = nn.Conv3d(dim, dim * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(dim * 4, dim, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return res + x

class GlobalUniBlock(nn.Module):
    """深层全局时空提取：在特征分辨率被降维后，执行全局多头自注意力"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # 将空间和时间维度展平为 Token 序列：[B, T*H*W, C]
        x_flat = x.flatten(2).transpose(1, 2) 
        
        # 全局时空注意力
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # 激活与映射
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # 恢复 3D 拓扑
        return x_flat.transpose(1, 2).reshape(B, C, T, H, W)
class Baseline_UniFormer_lite(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(Baseline_UniFormer_lite, self).__init__()
        
        # 1. 激进的空间降采样 Stem (256x256 -> 16x16)，彻底斩断显存危机
        self.stem = nn.Sequential(
            nn.Conv3d(2, cnn_channels, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)) # H, W 降维到 16
        )
        
        # 2. 浅层：堆叠 Local UniBlock 提取局部动态
        self.local_blocks = nn.Sequential(
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels)
        )
        
        # 3. 再次空间下采样 (16x16 -> 8x8)，为全局注意力准备
        self.downsample = nn.Sequential(
            nn.Conv3d(cnn_channels, hidden_size, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_size),
            nn.GELU()
        )
        
        # 4. 深层：Global UniBlock 执行全局时序与空间特征检索
        #self.global_block = GlobalUniBlock(hidden_size, num_heads=4)
        self.global_block = GlobalUniBlock(hidden_size, num_heads=2)
        # 5. 分类头
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, mask=None):
        # 跨模态张量转换: [B, T, 2, 256, 256] -> [B, 2, T, 256, 256]
        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # UniFormer 管线
        feat = self.stem(x_3d)
        feat = self.local_blocks(feat)
        feat = self.downsample(feat)
        feat = self.global_block(feat)
        
        # 压缩与分类
        pooled = self.gap(feat).flatten(1)
        return self.fc(pooled)


class Baseline_UniFormerV2_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(Baseline_UniFormerV2_5, self).__init__()
        
        # 1. 激进的空间降采样 Stem (256x256 -> 16x16)，斩断显存危机
        self.stem = nn.Sequential(
            nn.Conv3d(2, cnn_channels, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)) # H, W 降维到 16x16
        )
        
        # 2. 浅层：堆叠 Local UniBlock 提取局部动态 (在 16x16 尺度上)
        self.local_blocks = nn.Sequential(
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels)
        )
        
        # 3. 🌟 修改点：纯通道映射 (Channel Projection) 代替空间下采样
        # 空间分辨率保持 16x16 不变，仅通过 1x1x1 卷积将通道从 cnn_channels(32) 升至 hidden_size(64)
        self.channel_proj = nn.Sequential(
            nn.Conv3d(cnn_channels, hidden_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(hidden_size),
            nn.GELU()
        )
        
        # 4. 深层：Global UniBlock 执行全局时序与空间特征检索
        # ⚠️ 注意：此时喂给自注意力的 Token 数量是 16x16 (原来是 8x8)，即 256个/帧，保留了更多微观细节
        self.global_block = GlobalUniBlock(hidden_size, num_heads=2)
        
        # 5. 分类头
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, mask=None):
        # 跨模态张量转换: [B, T, 2, 256, 256] -> [B, 2, T, 256, 256]
        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # UniFormer 管线
        feat = self.stem(x_3d)
        feat = self.local_blocks(feat)
        feat = self.channel_proj(feat)  # 🌟 仅扩维，不降采样
        feat = self.global_block(feat)
        
        # 压缩与分类
        pooled = self.gap(feat).flatten(1)
        return self.fc(pooled)


class Baseline_UniFormerV3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(Baseline_UniFormerV3, self).__init__()
        
        # 1. 算力配平 Stem (256x256 -> 32x32)
        # 核心改动：第一步 Conv3d 仅使用 stride=2，保留 128x128 的高分辨率特征以吸收算力
        self.stem = nn.Sequential(
            nn.Conv3d(2, cnn_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.GELU(),
            # 第二步使用 stride=4 降采样到 32x32
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)) 
        )
        
        # 2. 浅层：扩展为 4 个 Local UniBlock
        # 核心改动：在 32x32 的分辨率上运行 4 层深度可分离卷积，极大丰富特征表达
        self.local_blocks = nn.Sequential(
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels)
        )
        
        # 3. 再次空间下采样 (32x32 -> 8x8)，为全局注意力准备，防止 OOM
        # 核心改动：为了匹配 8x8 的注意力池，此处 stride 设为 4，并增大感受野 kernel=5
        self.downsample = nn.Sequential(
            nn.Conv3d(cnn_channels, hidden_size, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(hidden_size),
            nn.GELU()
        )
        
        # 4. 深层：Global UniBlock (在 8x8 分辨率上执行全局自注意力，防爆显存)
        self.global_block = GlobalUniBlock(hidden_size, num_heads=2)
        
        # 5. 分类头
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, mask=None):
        # 跨模态张量转换: [B, T, 2, 256, 256] -> [B, 2, T, 256, 256]
        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        
        feat = self.stem(x_3d)
        feat = self.local_blocks(feat)
        feat = self.downsample(feat)
        feat = self.global_block(feat)
        
        pooled = self.gap(feat).flatten(1)
        return self.fc(pooled)


class Baseline_UniFormerV2_32x32(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(Baseline_UniFormerV2_32x32, self).__init__()
        
        # 1. 温和的空间降采样 Stem (256x256 -> 32x32)，保留更多微观形貌
        self.stem = nn.Sequential(
            nn.Conv3d(2, cnn_channels, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(cnn_channels),
            nn.GELU(),
            # 🌟 修改点：将 kernel_size 和 stride 从 (1, 4, 4) 改为 (1, 2, 2)
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # H, W 从 64 降维到 32
        )
        
        # 2. 浅层：此时 Local UniBlock 将在 32x32 的分辨率上提取特征
        self.local_blocks = nn.Sequential(
            LocalUniBlock(cnn_channels),
            LocalUniBlock(cnn_channels)
        )
        
        # 3. 再次空间下采样 (32x32 -> 16x16)
        self.downsample = nn.Sequential(
            # 注意：这里的 stride 依然是 (1, 2, 2)，所以输出变成了 16x16
            nn.Conv3d(cnn_channels, hidden_size, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(hidden_size),
            nn.GELU()
        )
        
        # 4. 深层：Global UniBlock 执行全局时序与空间特征检索
        self.global_block = GlobalUniBlock(hidden_size, num_heads=2)
        
        # 5. 分类头
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, mask=None):
        # 跨模态张量转换: [B, T, 2, 256, 256] -> [B, 2, T, 256, 256]
        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # UniFormer 管线
        feat = self.stem(x_3d)
        feat = self.local_blocks(feat)
        feat = self.downsample(feat)
        feat = self.global_block(feat)
        
        # 压缩与分类
        pooled = self.gap(feat).flatten(1)
        return self.fc(pooled)



class Aligned_Baseline_CNN_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super().__init__()
        
        # 1. 严格统一前端：使用与主推模型完全相同的联合降维编码器 (输入通道为 2，处理 PPI + STFT)
        self.shared_encoder = SymmetricDownsampleEncoder(in_channels=2, out_channels=cnn_channels)
        
        # 2. 空间特征降维：使用全局平均池化将 2D 空间特征图压缩为 1D 向量
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. 设定 Transformer 的嵌入维度（此时输入维度即为 cnn_channels）
        embed_dim = cnn_channels 
        
        # 4. 位置编码与 Transformer 编码器层
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, 
                                                    dim_feedforward=hidden_size * 4, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 5. 分类器
        self.fc = nn.Linear(embed_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W)
        B, T, C, H, W = x.size()
        
        # 合并 Batch 和 Time 维度送入共享 2D 编码器
        x_flat = x.view(B * T, C, H, W)
        
        # 提取 2D 空间特征: (B*T, cnn_channels, H_out, W_out)
        feat_2d = self.shared_encoder(x_flat)
        
        # 使用全局池化抹除空间二维光流特征，转换为 1D 序列: (B, T, cnn_channels)
        feat_1d = self.gap(feat_2d).view(B, T, -1)
        
        # 注入 Transformer 所需的时序位置编码
        trans_input = self.pos_encoder(feat_1d)
        
        # Transformer 时序建模
        trans_out = self.transformer(trans_input)
        
        # 对时间轴进行均值池化后分类输出
        return self.fc(self.dropout(trans_out.mean(dim=1)))






# ==========================================
# 🌟 普通ConvLSTM
# ==========================================

class SD_ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        # 注意去掉了 attn_heads 参数
        super(SD_ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. 联合降维编码器 (输入通道为 2: PPI + STFT)
        self.shared_encoder = SymmetricDownsampleEncoder(in_channels=2, out_channels=cnn_channels)
        
        # 2. 核心：纯粹的 ConvLSTM 单元 
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=cnn_channels if layer_idx == 0 else hidden_size,
                hidden_dim=hidden_size,
                kernel_size=3
            )
            for layer_idx in range(num_layers)
        ])
        
        # 3. 空间全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 分类器 (直接对接隐藏层，无自注意力，无维度翻倍)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W) 
        B, T, C, H, W = x.size()
        
        # 将 B 和 T 合并，进行 2D 卷积降采样
        x_flat = x.view(B * T, C, H, W)
        feat_2d = self.shared_encoder(x_flat) 
        
        # 恢复时间维度: (B, T, cnn_channels, H_out, W_out)
        _, C_out, H_out, W_out = feat_2d.size()
        feat_2d = feat_2d.view(B, T, C_out, H_out, W_out)
        
        # ConvLSTM 时空循环展开
        h_states = [
            torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device)
            for _ in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device)
            for _ in range(self.num_layers)
        ]
        
        temporal_feats = []
        for t in range(T):
            layer_input = feat_2d[:, t, :, :, :]
            for layer_idx, convlstm_cell in enumerate(self.convlstm_cells):
                h_states[layer_idx], c_states[layer_idx] = convlstm_cell(
                    layer_input,
                    (h_states[layer_idx], c_states[layer_idx])
                )
                layer_input = h_states[layer_idx]
            temporal_feats.append(h_states[-1])
            
        # 堆叠历史特征: (B, T, hidden, H_out, W_out)
        stacked_feats = torch.stack(temporal_feats, dim=1)
        
        # 使用全局平均池化压缩 2D 空间维度，保留时间维度
        # (B*T, hidden, H_out, W_out) -> (B*T, hidden, 1, 1) -> (B, T, hidden)
        seq_out = self.gap(stacked_feats.view(B * T, self.hidden_size, H_out, W_out)).view(B, T, self.hidden_size)
        
        # 🌟 最原始的时序聚合：直接对时间轴求平均 (彻底去除了 self_attn)
        pooled_features = seq_out.mean(dim=1)
        
        return self.fc(self.dropout(pooled_features))


class SingleStreamDownsampleEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        # 专职处理单模态 (纯 PPI 或纯 STFT)，彻底解耦物理特征
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 -> 64
            
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64 -> 16
        )
        
    def forward(self, x):
        return self.net(x)


class SE_ModalityFusion(nn.Module):
    def __init__(self, channels):
        super(SE_ModalityFusion, self).__init__()
        # channels 是单分支的输出维度 (例如 32)
        # 拼接后的总通道数为 channels * 2 (例如 64)
        in_channels = channels * 2
        
        # 降维比例 (Reduction Ratio) 设为 4，极限压缩算力
        reduced_channels = max(8, in_channels // 4)
        
        # 动态权重生成网络
        self.se_weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # Squeeze: 提取全局空间与频率的极值上下文
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()                           # Excitation: 输出 0~1 之间的动态权重门控
        )

    def forward(self, feat_ppi, feat_stft):
        # feat_ppi, feat_stft shape: (B*T, channels, H, W)
        
        # 1. 物理特征初步对齐拼接
        concat_feat = torch.cat([feat_ppi, feat_stft], dim=1)  # shape: (B*T, 2*channels, H, W)
        
        # 2. 生成通道级自适应权重
        weights = self.se_weight_generator(concat_feat).view(concat_feat.size(0), -1, 1, 1)
        
        # 3. 模态重加权 (Modality Reweighting)
        # 如果当前帧机动剧烈，STFT 通道的权重会被网络自动放大；反之则放大 PPI 通道
        fused_feat = concat_feat * weights
        
        return fused_feat

class TwoStream_Adaptive_ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(TwoStream_Adaptive_ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_channels = cnn_channels
        
        # 1. 🌟 双分支独立特征解耦提取器
        self.encoder_ppi = SingleStreamDownsampleEncoder(in_channels=1, out_channels=cnn_channels)
        self.encoder_stft = SingleStreamDownsampleEncoder(in_channels=1, out_channels=cnn_channels)
        
        # 2. 🌟 自适应模态融合模块
        self.adaptive_fusion = SE_ModalityFusion(channels=cnn_channels)
        
        # 3. 时空循环动力学建模
        fusion_out_channels = cnn_channels * 2  # 拼接后的总维度
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=fusion_out_channels if layer_idx == 0 else hidden_size,
                hidden_dim=hidden_size,
                kernel_size=3
            )
            for layer_idx in range(num_layers)
        ])
        
        # 4. 空间全局平均池化与分类头
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W)
        B, T, C, H, W = x.size()
        
        # 1. 严格分离物理模态
        ppi = x[:, :, 0:1, :, :].contiguous().view(B * T, 1, H, W)
        stft = x[:, :, 1:2, :, :].contiguous().view(B * T, 1, H, W)
        
        # 2. 双流独立降维
        feat_ppi = self.encoder_ppi(ppi)
        feat_stft = self.encoder_stft(stft)
        
        # 3. 模态自适应动态融合
        # fused_feat shape: (B*T, 2*cnn_channels, H_out, W_out)
        fused_feat = self.adaptive_fusion(feat_ppi, feat_stft)
        
        # 恢复时间维度进行时空循环
        _, C_out, H_out, W_out = fused_feat.size()
        fused_feat = fused_feat.view(B, T, C_out, H_out, W_out)
        
        # 4. ConvLSTM 前向传播
        h_states = [torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device) for _ in range(self.num_layers)]
        
        temporal_feats = []
        for t in range(T):
            layer_input = fused_feat[:, t, :, :, :]
            for layer_idx, convlstm_cell in enumerate(self.convlstm_cells):
                h_states[layer_idx], c_states[layer_idx] = convlstm_cell(
                    layer_input,
                    (h_states[layer_idx], c_states[layer_idx])
                )
                layer_input = h_states[layer_idx]
            temporal_feats.append(h_states[-1])
            
        stacked_feats = torch.stack(temporal_feats, dim=1)
        
        # 5. 特征压缩与分类 (极简池化)
        seq_out = self.gap(stacked_feats.view(B * T, self.hidden_size, H_out, W_out)).view(B, T, self.hidden_size)
        pooled_features = seq_out.mean(dim=1)
        
        return self.fc(self.dropout(pooled_features))




class DualStream_ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32):
        super(DualStream_ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. 双分支独立编码器 (注意：输入必须是 1 通道)
        self.encoder_ppi = SingleStreamDownsampleEncoder(in_channels=1, out_channels=cnn_channels)
        self.encoder_stft = SingleStreamDownsampleEncoder(in_channels=1, out_channels=cnn_channels)
        
        # 2. 时空循环动力学建模
        # 核心：融合后的通道数为 cnn_channels * 2 (例如 32+32=64 通道)
        fusion_out_channels = cnn_channels * 2  
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=fusion_out_channels if layer_idx == 0 else hidden_size,
                hidden_dim=hidden_size,
                kernel_size=3
            )
            for layer_idx in range(num_layers)
        ])
        
        # 3. 空间全局平均池化与分类头
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W)
        B, T, C, H, W = x.size()
        
        # 1. 严格分离物理模态
        ppi = x[:, :, 0:1, :, :].contiguous().view(B * T, 1, H, W)
        stft = x[:, :, 1:2, :, :].contiguous().view(B * T, 1, H, W)
        
        # 2. 双流独立降维 (保留二维空间结构，绝不展平！)
        # 输出形状: (B*T, cnn_channels, H_out, W_out)
        feat_ppi = self.encoder_ppi(ppi)
        feat_stft = self.encoder_stft(stft)
        
        # 3. 经典的通道拼接融合 (Channel Concatenation)
        # 融合后形状: (B*T, 2*cnn_channels, H_out, W_out)
        fused_feat = torch.cat([feat_ppi, feat_stft], dim=1)
        
        # 恢复时间维度进行时空循环
        _, C_out, H_out, W_out = fused_feat.size()
        fused_feat = fused_feat.view(B, T, C_out, H_out, W_out)
        
        # 4. ConvLSTM 前向传播
        h_states = [torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device) for _ in range(self.num_layers)]
        
        temporal_feats = []
        for t in range(T):
            layer_input = fused_feat[:, t, :, :, :]
            for layer_idx, convlstm_cell in enumerate(self.convlstm_cells):
                h_states[layer_idx], c_states[layer_idx] = convlstm_cell(
                    layer_input,
                    (h_states[layer_idx], c_states[layer_idx])
                )
                layer_input = h_states[layer_idx]
            temporal_feats.append(h_states[-1])
            
        stacked_feats = torch.stack(temporal_feats, dim=1)
        
        # 5. 特征压缩与分类 (极简池化)
        seq_out = self.gap(stacked_feats.view(B * T, self.hidden_size, H_out, W_out)).view(B, T, self.hidden_size)
        pooled_features = seq_out.mean(dim=1)
        
        return self.fc(self.dropout(pooled_features))










class OURS_ST_DLAN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, cnn_channels=32, attn_heads=2):
        super(OURS_ST_DLAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1 # ConvLSTM 通常单向即可捕捉前向光流
        
        # 1. 联合降维编码器 (输入通道为 2: PPI + STFT)
        self.shared_encoder = SymmetricDownsampleEncoder(in_channels=2, out_channels=cnn_channels)
        
        # 2. 核心：纯粹的 ConvLSTM 单元 (需要复用上一轮提供的 ConvLSTMCell 类)
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=cnn_channels if layer_idx == 0 else hidden_size,
                hidden_dim=hidden_size,
                kernel_size=3
            )
            for layer_idx in range(num_layers)
        ])
        
        # 3. 空间全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 时序自注意力层
        self.self_attn = MultiHeadSelfAttention_noresidual(embed_dim=hidden_size, num_heads=attn_heads, dropout=dropout)
        
        # 5. 分类器 (🌟 修改点 1：传统池化，输入维度恢复为 hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (B, T, 2, H, W) 
        B, T, C, H, W = x.size()
        
        # 将 B 和 T 合并，进行 2D 卷积降采样
        x_flat = x.view(B * T, C, H, W)
        feat_2d = self.shared_encoder(x_flat) 
        
        # 恢复时间维度: (B, T, cnn_channels, H_out, W_out)
        _, C_out, H_out, W_out = feat_2d.size()
        feat_2d = feat_2d.view(B, T, C_out, H_out, W_out)
        
        # ConvLSTM 时空循环展开
        h_states = [
            torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device)
            for _ in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(B, self.hidden_size, H_out, W_out, device=x.device)
            for _ in range(self.num_layers)
        ]
        
        temporal_feats = []
        for t in range(T):
            layer_input = feat_2d[:, t, :, :, :]
            for layer_idx, convlstm_cell in enumerate(self.convlstm_cells):
                h_states[layer_idx], c_states[layer_idx] = convlstm_cell(
                    layer_input,
                    (h_states[layer_idx], c_states[layer_idx])
                )
                layer_input = h_states[layer_idx]
            temporal_feats.append(h_states[-1])
            
        # 堆叠历史特征: (B, T, hidden, H_out, W_out)
        stacked_feats = torch.stack(temporal_feats, dim=1)
        
        # 使用全局平均池化压缩 2D 空间维度，保留时间维度
        # (B*T, hidden, H_out, W_out) -> (B*T, hidden, 1, 1) -> (B, T, hidden)
        seq_out = self.gap(stacked_feats.view(B * T, self.hidden_size, H_out, W_out)).view(B, T, self.hidden_size)
        
        # 自注意力机制寻找关键帧
        attn_out = self.self_attn(seq_out, mask=mask)
        
        # 🌟 修改点 2：传统均值池化 (直接对时间序列求平均，不再做 Max-Mean 拼接)
        pooled_features = attn_out.mean(dim=1)
        
        return self.fc(self.dropout(pooled_features))

# ==========================================
# 辅助函数区
# ========================================== 
def create_model(model_type, input_size, hidden_size, num_layers, output_size, dropout, cnn_channels, kernel_size,
                 attn_heads=4):
    #基线模型
    
    if model_type == 'BASELINE_GRU':
        return Baseline_GRU(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels)
    if model_type == 'BASELINE_LSTM':
        return Baseline_LSTM(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels)
    if model_type == 'BASELINE_BIGRU_SA':
        return Baseline_BiGRU_SA(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels, attn_heads)
    
    ###主推的新模型
    if model_type == 'SD_ConvLSTM':
        return SD_ConvLSTM(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels)
    if model_type == 'ST-DLAN':
        return OURS_ST_DLAN(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels)
    if model_type == 'uniformer-lite':
        return Baseline_UniFormer_lite(input_size, hidden_size, num_layers, output_size, dropout, cnn_channels)


    else:
        raise ValueError(f"当前暂不支持转换为 2D 的模型: {model_type}")


def train_and_recognize_model(train_loader, val_loader, test_loader, num_classes, model_type='C_CNNBIGRU_SA', 
                              num_epochs=100, learning_rate=0.001,
                              hidden_size=64, num_layers=2, dropout=0.2, cnn_channels=64, kernel_size=3,
                              attn_heads=4,
                              dataset_name="Dataset", dataset_size=None, logger=None):
    
    model = create_model(
        model_type, 1, hidden_size, num_layers, num_classes, dropout, cnn_channels, kernel_size,
        attn_heads=attn_heads
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 1. 优化器升级：加入 weight_decay=5e-4 (L2 正则化)，强行压制过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # 2. 调度器升级：抛弃阶梯突降，使用丝滑的余弦退火 (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # =========================================================
    # 🌟 算力与参数量探针 (基于真实 DataLoader 样本)
    # =========================================================
    flops_str, params_str = "N/A", "N/A"
    try:
        from thop import profile, clever_format
        sample_seq, _ = next(iter(val_loader))
        dummy_input = sample_seq[0:1].to(device).float() / 255.0  
        
        model.eval()
        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
            flops_str, params_str = clever_format([macs * 2, params], "%.3f")
            
        if logger is not None:
            logger.info(f"🚀 【算力评估】 模型: {model_type} | FLOPs: {flops_str} | Params: {params_str}")
        print(f"\n🚀 【算力评估】 FLOPs: {flops_str} | Params: {params_str}\n")
    except Exception as e:
        print(f"\n⚠️ 算力计算失败: {e}\n")
    # =========================================================

    # =========================================================
    # 🌟 权重保存路径初始化 (文件名标记为 last_epoch)
    # =========================================================
    weights_dir = os.path.join("results", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    size_part = dataset_size_tag(dataset_size) if dataset_size else ""
    weight_filename = f"{safe_name(dataset_name)}_{safe_name(model_type)}_L{num_layers}_H{attn_heads}_{size_part}_{time_str}_last_epoch.pth"
    last_weight_path = os.path.join(weights_dir, weight_filename)

    with tqdm(total=num_epochs, desc=f"Training {model_type}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            total_train_loss = 0
            train_correct, train_total = 0, 0

            for sequences, labels in train_loader:
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                sequences = sequences.float() / 255.0
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += loss.item()

                _, predicted = torch.max(outputs.detach(), 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            model.eval()
            total_val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    sequences = sequences.float() / 255.0
                    
                    with autocast():
                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                        
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct / total

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_elapsed_time = time.time() - epoch_start_time

            if logger is not None:
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Epoch Elapsed Time: {epoch_elapsed_time:.2f} s | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.3f}",
                "Train Acc": f"{train_accuracy:.3f}",
                "Val Loss": f"{avg_val_loss:.3f}",
                "Val Acc": f"{val_accuracy:.3f}"
            })
            pbar.update(1)
            scheduler.step()
            
    # =========================================================
    # 🌟 核心修改：在所有 Epoch 结束后，直接保存当前（即最后一轮）的模型权重
    # =========================================================
    torch.save(model.state_dict(), last_weight_path)
    if logger is not None:
        logger.info(f"✅ Training completed! Successfully saved LAST epoch model weights to: {last_weight_path}")
        
    del train_loader
    del val_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 测试与验证逻辑（此时 model 就是最后一轮的状态，直接前向传播即可）
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sequences = sequences.float() / 255.0
            
            with autocast():
                outputs = model(sequences)
                
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    plot_confusion_matrix(
        all_labels, all_preds,
        [str(i) for i in np.unique(all_labels)],
        model_type,
        accuracy, precision, recall, f1,
        dataset_name=dataset_name,
        dataset_size=dataset_size
    )
     
    return model, accuracy, precision, recall, f1, flops_str, params_str

def setup_logger(save_dir, logger_name="experiment_logger",
                 dataset_name=None, model_name=None, dataset_size=None):
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    time_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    parts = ["training"]

    if dataset_name is not None:
        parts.append(safe_name(dataset_name))
    if model_name is not None:
        parts.append(safe_name(model_name))
    if dataset_size is not None:
        parts.append(dataset_size_tag(dataset_size))

    parts.append(time_str)
    log_file = os.path.join(log_dir, "_".join(parts) + ".log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def plot_confusion_matrix(y_true, y_pred, classes, net_name,
                          accuracy, precision, recall, f1, dataset_name="Dataset",
                          dataset_size=None,
                          save_path='results/confusion_matrix/', 
                          figsize=(18, 16)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 8})
                
    plt.xlabel("Predicted Label (0-13)", fontsize=13)
    plt.ylabel("True Label (0-13)", fontsize=13)
    plt.title(f"14-Class Intention Confusion Matrix - {net_name} ({dataset_name})", fontsize=16)
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}   "
        f"Precision: {precision:.4f}   "
        f"Recall: {recall:.4f}   "
        f"F1-score: {f1:.4f}"
    )
    plt.figtext(0.5, 0.02, metrics_text, ha="center", fontsize=14, weight='bold')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        time_str = datetime.now().strftime("%H-%M-%S")

        size_part = f"_{dataset_size_tag(dataset_size)}" if dataset_size is not None else ""
        filename = f"{safe_name(dataset_name)}_{safe_name(net_name)}_14classes_cm{size_part}_{time_str}.png"

        plt.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":

    # ==========================================
    # 🌟 核心升级：HDF5 超级文件数组遍历！ 🌟
    # ==========================================
    DATASET_H5_FILES = [
        #'0904_Dataset_14Classes_SNR1000_RCS20_T-100to100_Step10_Num105000.h5',
        #'Dataset_14Classes_SNR10_RCS20_T-100to100_Step10_Num500.h5',
        'Dataset_14Classes_SNR1000_RCS20_T-100to100_Step10_Num105000_0.h5',
        'Dataset_14Classes_SNR10_RCS20_T-100to100_Step10_Num105000_0.h5',
        

    ]
        
    
    # 🌟 包含 4 个基线模型 + 4 个消融模型的完整列表
    config_1 = [


    # {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'Ours_ST-DLAN', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':16, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # },  
    #     {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'Ours_ST-DLAN', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 
    #     {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'Ours_ST-DLAN', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':64, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 
    #     {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'Ours_ST-DLAN', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':128, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 



    # {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'ST-DLAN', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 

    {
               'save_dir': 'results', 'model_configs': { 
                   'default': 'BASELINE_BIGRU_SA', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
                   'attn_heads': 1
               },
               'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    }, 



    # {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'Baseline_CNN_ConvLSTM', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 
    # {
    #            'save_dir': 'results', 'model_configs': { 
    #                'default': 'uniformer-lite', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                'attn_heads': 2
    #            },
    #            'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # },
    # {
    #                'save_dir': 'results', 'model_configs': { 
    #                    'default': 'BASELINE_GRU', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                    'attn_heads': 2
    #                },
    #                'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 
    # {
    #                'save_dir': 'results', 'model_configs': { 
    #                    'default': 'BASELINE_LSTM', 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3, 'cnn_channels':32, 'kernel_size':3,
    #                    'attn_heads': 2
    #                },
    #                'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001}
    # }, 
    
    ] #后面只保留cnn-transformer 、cnn-gru 、 baseline-convlstm 


    
    

    all_results = []
    
    # ========================================== 
    # 🌟 极速外层循环：遍历所有 HDF5 数据集
    # ==========================================
    for current_h5_file in DATASET_H5_FILES:
        # 提取纯净的数据集名称 (去除扩展名) 用于打印和图表保存
        dataset_name = os.path.basename(current_h5_file).replace('.h5', '')
        
        print(f"\n=======================================================================================")
        print(f"🚀 开启全新 HDF5 数据集测试流程: 【 {dataset_name} 】")
        print(f"=======================================================================================")
        



        # 1. 实例化 HDF5 Dataset
        full_dataset = FastHDF5Dataset(current_h5_file)
        num_classes = full_dataset.num_classes
        total_len = len(full_dataset)

        train_size = int(total_len * (4 / 6))  # 占 4/6 (约 66.67%)
        val_size = int(total_len * (1 / 6))    # 占 1/6 (约 16.67%)
        test_size = total_len - train_size - val_size  # 剩下的自动划给测试集

        # 🌟 快速读取一次全体标签，用于作为分层抽样的依据
        print("🔍 正在提取全局标签进行严谨的分层抽样 (Stratified Split)...")
        with h5py.File(current_h5_file, 'r') as f:
            all_labels = f['labels'][:]

        # 2. 绝对平衡的数据集切分比例 (4/6, 1/6, 1/6)
        # 2. 绝对平衡的数据集切分比例 (4/6, 1/6, 1/6)
        indices = np.arange(total_len)
        
        # =========================================================
        # 🚀 恢复严格的数学分层抽样 (Stratified Split)
        # =========================================================
        # 第一次切分：分离出 训练集 (4/6) 和 临时集 (2/6)
        train_idx, temp_idx, train_labels_subset, temp_labels = train_test_split(
            indices, all_labels, 
            test_size=(2/6), 
            stratify=all_labels, 
            random_state=42
        )
        
        # 第二次切分：将临时集平分为 验证集 (1/6) 和 测试集 (1/6)
        val_idx, test_idx, val_labels_subset, test_labels_subset = train_test_split(
            temp_idx, temp_labels,
            test_size=0.5, 
            stratify=temp_labels, 
            random_state=42
        )

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(test_dataset)

        print(f"✅ 【{dataset_name}】 严格分层切分完毕！训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

        # 2. 统计并打印真实分布，验证绝对均匀性
        print("\n📊 --- 严格分层抽样下的标签分布统计 ---")
        print(f"{'类别 (Class)':<12} | {'训练集 (Train)':<12} | {'验证集 (Val)':<12} | {'测试集 (Test)':<12}")
        print("-" * 65)
        
        for c in range(num_classes):
            tr_count = np.sum(train_labels_subset == c)
            va_count = np.sum(val_labels_subset == c)
            te_count = np.sum(test_labels_subset == c)
            print(f"Class {c:<6} | {tr_count:<14} | {va_count:<12} | {te_count:<12}")
            
        print("-" * 65)
        print(f"{'Total':<12} | {train_size:<14} | {val_size:<12} | {test_size:<12}")
        print("-----------------------------------------------------------\n")
        # =========================================================

        
        # 3. 开始遍历配置进行实验
        for i in range(len(config_1)):
            seed_everything(42)
            cfg = config_1[i]
            attn_heads = cfg['model_configs'].get('attn_heads', 4)
            experiment_model_name = (
                f"{cfg['model_configs']['default']}_layers{cfg['model_configs']['num_layers']}"
                f"_heads{attn_heads}"
            )
            logger = setup_logger(cfg['save_dir'],dataset_name=dataset_name,model_name=experiment_model_name,dataset_size=total_len)
            logger.info("---------------------------------------------------------")
            logger.info("📋 实验超参数与模型配置详情 (Experiment Configurations):")
            # 将 cfg 字典格式化为带缩进的 JSON 格式字符串
            cfg_pretty_str = json.dumps(cfg, indent=4, ensure_ascii=False)
            for line in cfg_pretty_str.split('\n'):
                logger.info(line)
            logger.info("=========================================================")
            logger.info(f"在数据集 【{dataset_name}】 上测试模型: {cfg['model_configs']['default']} | num_layers={cfg['model_configs']['num_layers']}")
            batch_size = cfg['train_params']['batch_size']
            
            # 使用 HDF5 后，可以适当开启多个 worker 榨干 I/O
            #num_workers = max(2, multiprocessing.cpu_count() - 4)
            # 修改后：
            num_workers = 4

            # 🌟 1. 创建一个严格绑定的随机生成器
            g = torch.Generator()
            g.manual_seed(42)

            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,                # 🌟 核心：恢复洗牌！无压缩数据不怕随机读！
                num_workers=num_workers,     # 🌟 提升至 4 进程
                pin_memory=True,             # 🌟 恢复锁页内存，提速数据进显卡的通道
                prefetch_factor=2, 
                persistent_workers=True,
                worker_init_fn=seed_worker,  
                generator=g                  
            )

            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers, 
                pin_memory=True, 
                persistent_workers=True,
                worker_init_fn=seed_worker   # 验证集不 shuffle，但也加上确保安全
            )

            # test_loader = DataLoader(
            #     test_dataset, 
            #     batch_size=batch_size, 
            #     shuffle=False, 
            #     num_workers=num_workers, 
            #     pin_memory=True,
            #     worker_init_fn=seed_worker
            # )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0,       # 强制单进程
                pin_memory=True,
                worker_init_fn=seed_worker
            )



            # num_workers = 4
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
            # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            start_time = time.time()
            
            # 🌟 接收新增的 flops 和 params
            model, accuracy, precision, recall, f1, flops, params = train_and_recognize_model(
                train_loader, val_loader, test_loader, num_classes,
                model_type      = cfg['model_configs']['default'],
                num_epochs      = cfg['train_params']['epochs'],
                learning_rate   = cfg['train_params']['learning_rate'],
                hidden_size     = cfg['model_configs']['hidden_size'],
                num_layers      = cfg['model_configs']['num_layers'],
                dropout         = cfg['model_configs']['dropout'],
                cnn_channels    = cfg['model_configs']['cnn_channels'],
                kernel_size     = cfg['model_configs']['kernel_size'],
                attn_heads      = attn_heads,
                dataset_name    = dataset_name,
                dataset_size    = total_len,
                logger          = logger  
            )
            
            elapsed_time = round(time.time() - start_time, 2)
            
            # 🌟 记录结果，新增 FLOPs 和 Params 列
            all_results.append({
                "Dataset": dataset_name,
                "Model": cfg['model_configs']['default'],
                "Num_layers": cfg['model_configs']['num_layers'],
                "Attn_heads": attn_heads,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Train_size": train_size,
                "Test_size": test_size,
                "Time(s)": elapsed_time,
                "FLOPs": flops,       # 记录计算复杂度
                "Params": params      # 记录模型参数量
            })

            # 强制垃圾回收和显存释放
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # === 所有数据集循环结束后，生成超级汇总表格 ===
    results_df = pd.DataFrame(all_results)
    print("\n=======================================================================================")
    print("🎯 所有数据集 & 所有雷达模型性能超级汇总表")
    print("=======================================================================================")
    print(results_df.to_string())
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    size_part = dataset_size_tag(total_len)
    csv_filename = f"Super_Summary_HDF5_14classes_performance_{size_part}_{time_str}.csv"
    results_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n📊 汇总结果已成功保存至: {csv_filename}")
