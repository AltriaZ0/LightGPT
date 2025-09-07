from data import TextDataProcessor
from nvtx import nvtx_range
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码
class PositionalEncoding(nn.Module):
    """
    实现Transformer中的固定位置编码
    生成后在整个训练过程中保持不变
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        :param d_model: 嵌入维度
        :param dropout: dropout概率
        :param max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).cuda()
        
        # 计算频率项 (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model)).cuda()
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term).cuda() # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term).cuda() # 奇数索引使用cos
        
        # 增加batch维度 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    @nvtx_range()
    def forward(self, x):
        """
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :return: 添加位置编码后的张量
        """
        # 添加位置编码（只取前x.size(1)个位置）
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(dtype=x.dtype)
        return self.dropout(x)
    
# 多头注意力 优化版本
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout_p  = float(dropout)

        # 一次线性得到 QKV（比逐头 Linear 高效得多）
        self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # 输出投影
        self.proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @nvtx_range()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # 1) 一次性得到 QKV，并 reshape 成 (B, H, T, Dh)
        with torch.cuda.nvtx.range("qkv_linear"):
            qkv = self.W_qkv(x)                                     # (B, T, 3C)
            q, k, v = qkv.chunk(3, dim=-1)                          # 三份 (B, T, C)
            # 变成 (B, H, T, Dh)，注意先 contiguous 再 view/reshape
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2) SDPA：所有头在一个/两个大核里并行完成
        p = self.dropout_p if self.training and self.dropout_p > 0 else 0.0
        AMP_DTYPE = torch.bfloat16  # Ada/Lovelace 推荐 BF16；也可用 torch.float16
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            with torch.cuda.nvtx.range("SDPA(causal)"):
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=p,
                    is_causal=True
                )                                                    # (B, H, T, Dh)

            # 3) 合并头并做输出投影
            with torch.cuda.nvtx.range("proj"):
                out = attn_out.transpose(1, 2).reshape(B, T, C)          # (B, T, C)
                out = self.dropout(self.proj(out))                       # (B, T, C)
        return out

#  前馈网络
class FeedForward(nn.Module):
    def __init__(self, Embedding_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Embedding_dim, Embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(Embedding_dim * 4, Embedding_dim),
            nn.Dropout(dropout)
        )

    @nvtx_range()
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, Embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(Embedding_dim, num_heads) # 自注意力
        self.ffwd = FeedForward(Embedding_dim, dropout=0.1) # 前向传播
        self.ln1 = nn.LayerNorm(Embedding_dim) # 层归一化
        self.ln2 = nn.LayerNorm(Embedding_dim)

    @nvtx_range()
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Pre-Norm: 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x))  # Pre-Norm：残差前向传播层
        return x

# 语言模型
class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 block_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim  = embed_dim
        self.num_heads = num_heads
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=0.1, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, dropout) for _ in range(num_layers)]) # 脚注⑤
        self.ln_f = nn.LayerNorm(embed_dim) # 最后的层归一化
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # 权重共享：lm_head weight 与 embedding weight 绑定
        # todo：研究
        if tie_weights:
            self.lm_head.weight = self.token_embedding_table.weight

        # 参数初始化（GPT 风格，兼容 sdpa）
        # todo：研究
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @nvtx_range()
    def forward(self, idx, targets=None):
        B, T = idx.shape
        with torch.cuda.nvtx.range("词嵌入"):
            token_embd = self.token_embedding_table(idx)  # (B, T, Embedding_dim)
        with torch.cuda.nvtx.range("位置编码"):
            x = self.positional_encoding(token_embd)  # (T, Embedding_dim)
       
        x = self.blocks(x)  # (B, T, Embedding_dim)

        with torch.cuda.nvtx.range("Final and Linear"):
            x = self.ln_f(x)  # (B, T, Embedding_dim)
            logits = self.lm_head(x)

        if targets is not None:
            logits = logits.view(B * T, self.vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) #交叉熵
        else:
            loss = None
        return logits, loss
    
    @torch.no_grad()     
    @nvtx_range()
    def generate(self, idx, max_new_tokens=100):
        """
        输入: idx (B, T) 的起始序列
        输出: 生成后的序列 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 截取上下文窗口
            idx_cond = idx[:, -self.block_size:]
            # 前向传播
            logits, _ = self(idx_cond)
            # 取最后一个时间步
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # 采样
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接
            idx = torch.cat((idx, next_token), dim=1)
        return idx


