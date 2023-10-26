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
    
class RMSNorm(nn.Module):
    """
    rms_norm(x) = x/sqrt(mean(xi**2))
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
    

