import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    """
    Params of the model
    """
    dim: int = 4096 # embedding dim
    n_layers: int = 32 #num attn blocks
    n_heads: int = 32 #num heads for queries
    n_kv_heads: Optional[int] = None #num heads for K and V
    vocab_size: int = -1 #set in tokenizer
    multiple_of: int = 256 #hidden_dim of FFN (?)
    ffn_dim_multiplier: Optional[float] = None #hidden_dim of FFN(?)
    norm_eps: float = 1e-5
    
    #KV cache related
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(
    head_dim: int, #embedding dim
    seq_len: int, #max length of seq, here: max_seq_len*2
    device: str,
    theta: float = 10000.0 #theta value in RoPE
):
    assert head_dim % 2 == 0, "embedding dimension must be divisible by 2" #rope paper constraint
    
    #build theta params according to the formula in the rope paper: 
    # theta_i = 10000 ^ (-2(i-1)/dim for i in [1, 2, ....dim/2]) #[0, 2, ....]
    #shape: (head_dim/2)
    theta_numerator = torch.arange(1, head_dim, 2).float() #[1.0, 3.0, 5.0, ....head_dim-1]??
    theta = 1.0/(theta ** (theta_numerator/head_dim)).to(device)
    #construct the positions (`m` param). shape: (max_seq_len)
    m = torch.arange(max_seq_len, device=device)
    #outer product: multiply each theta by each position using outer product.
    #shapes: (seq_len) outer_prod (head_dim//2) -> (seq_len, head_dim//2)
    freqs = torch.outer(m, theta).float()
    #we can compute complex numbers in the polar form: c = R * exp(i*m*theta), where R=1 as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    

def apply_rotary_embeddings(x: torch.Tensor,
                            freqs_complex:torch.Tensor,
                            device:str
                            ):
    
    #x is the token already divided by num heads for multi-headed self attention.
    #transform x: [x1, x2, x3, x4] -> [[x1, x2], [x3, x4]] -> [x1 + ix2, x3+ix4]
    #shape: (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    #(seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2): add batch and head dim
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #element-wise multiplication with freqs_complex rotates x
    #shape: (B, seq_len,H,head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    #[x1+ix2, x3+ix4] -> [[x1,x2],[x3,x4]]
    #(B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    #(B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return(
            #(B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim) #make num kv heads = num q heads
            .reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim) #flatten
        )
        
class RMSNorm(nn.Module):
    """
    rms_norm(x) = gamma * x/sqrt(mean(xi**2))
    where gamma is a learnable param which controls the amount of re-scaling invariance.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        #gamma param in RMS norm
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x:torch.Tensor):
        #(B, seq_len, dim)
        #rqsrt = 1/(sqrt(x))
        return x*torch.rqsrt(x.pow(2).mean(-1, keepdim=True)+ self.eps)
    
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        #(dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    
    def __init__(self, args:ModelArgs):
        super().__init__()
        
        #number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #number of heads for Queries
        self.n_q_heads = args.n_heads
        #ratio of number of heads of Queries to number of KV heads (number of times KV heads must be repeated to get Q heads)
        self.n_rep = self.n_q_heads // self.n_kv_heads
        #dimension of each head in multi-head self attention.
        self.head_dim = args.dim // args.n_heads
        
        #input token has size dim, output query token has n_head different representations and each has size head_dim
        self.wq = nn.Linear(args.dim, args.n_q_heads*self.head_dim, bias=False)
        #input token has size dim, output key token has n_kv_heads (<=n_heads) rep and each has size head_dim
        self.wk = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        #input token has size dim, output value token has n_kv_heads rep and each of these has size head_dim
        self.wv = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        #input token is the output of attention and has size of (n_heads, head_dim)
        #output token expected size is: (dim)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)
        
        #KV cache
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape() #(B, 1, dim)
        
        #multiply the input emb with WQ, WK, WV to get Q, K, V.
        #(B, 1, dim) -> (B, 1, H_Q*head_dim)
        xq = self.wq(x)
        #(B, 1, dim) -> (B,1, H_KV*head_dim)
        xk = self.wk(x)
        #(B, 1, dim) -> (B, 1, H_KV*head_dim)
        xv = self.wv(x)
        
        #(B, 1, HQ*head_dim) -> (B, 1, HQ, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        #(B, 1, HKV*head_dim) -> (B, 1, HKV, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        #(B, 1, HKV*head_dim) -> (B, 1, HKV, head_dim)
        xv = xv.view(batch_size, seq_len, n_kv_heads, self.head_dim)
        
        #apply RoPE to Q, K only. (output is same size as input)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        #replace the entry in the KV cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        #retreive all cached keys and values so far for attention calculation
        #(B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size,0:start_pos+seq_len] #?
        values = self.values_k[:batch_size,0:start_pos+seq_len] #?
        
        #repeat the heads of K and V to equal number of queries (not-optimal)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        #multi-head attention
        #(B, 1, H_Q, Head_Dim) --> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        #(B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) --> (B, HQ, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        #(B, HQ, 1, Seq_Len_KV) @(B, HQ, Seq_Len_KV, Head_Dim) --> (B, HQ, 1, Head_Dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        
        return output
    
class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        #normalization before attention
        self.attention_norm = RMSNorm(args.dim, eps=args.eps)
        #self attention block
        self.attention = SelfAttention(args)
        #normalization before the feedforward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.eps)
        #feed forward block
        self.feed_forward = FeedForward(args)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        #(B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out
        
class Transformer(nn.Module):
    """
    A class the implements the entire model architecture:
    EMB -> (RMS Norm -> Self-Attention block -> Residual -> RMS Norm -> FFN+SwiGLU) * n_layers -> RMS Norm -> Linear
    """
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "set the vocab size in model args!"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim) # V X D embedding lookup
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args)
        self.ffn(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False) # D X V

        #rotary positional embeddings
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.num_heads, #?
            self.args.max_seq_len*2, #?
            device=self.args.device
            )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> nn.Module:

        #(B, Seq_Len)
        batch_size, seq_len = tokens.shape
        #input is only 1 token since all previous tokens will be kept in KV cache
        assert seq_len == 1, "Only 1 token can be processed at a time!"
        
        #(B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        #retreive pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        #encode input with Encoder
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float() #why .float()?

        return output