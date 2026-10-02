import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, max_len=1000, dropout=0.3, debug=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative bias: [2*max_len-1, num_heads]
        self.rel_bias = nn.Embedding(2 * max_len - 1, num_heads)
        self.debug = debug

    def forward(self, x, mask=None):
        B, T, D = x.shape

        residual = x
        x = self.norm(x)

        # 1. QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, T, head_dim]

        # 2. Scaled Dot-Product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, T, T]

        
        # 3. Add Relative Bias BEFORE Softmax
        pos = torch.arange(T, device=x.device)
        rel_dist = pos[None, :] - pos[:, None]  # [T, T]
        max_rel = (self.rel_bias.num_embeddings - 1) // 2
        if self.debug:
            print("T:", T)
            print("max_rel:", max_rel)
            print("rel_dist min/max:", rel_dist.min().item(), rel_dist.max().item())
        # Clap relative distances to the range [-max_rel, max_rel]
        rel_dist = rel_dist.clamp(-max_rel,max_rel)
        rel_idx = rel_dist + max_rel
        # [T, T, heads] -> [heads, T, T]
        bias = self.rel_bias(rel_idx).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        
        if mask is not None:
            if self.debug:
                print("mask sum:", mask.sum(dim=1))
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn = attn.masked_fill(~mask_expanded, float('-inf'))

        # 4. Finalize Attention
        attn = attn.softmax(dim=-1)
        if mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
        if self.debug and mask is not None:
            # Let's inspect Batch 0, Head 0, Query Row 0
            # We print all columns (Keys) for this specific row
            sample_mask_sum = mask[0].sum().item()
            print(f"\n--- Attention Mask Verification (T={T}) ---")
            print(f"Sample 0 true frame count: {sample_mask_sum}")
            
            # Extract row 0 after softmax
            row_slice = attn[0, 0, 0, :] 
            print("Softmax row values:\n", row_slice.detach().cpu().numpy())
            
            # Check if padding columns are strictly 0.0
            padding_start = sample_mask_sum
            if padding_start < T:
                pad_vals = row_slice[padding_start:]
                print(f"Values in padding columns (idx {padding_start} to {T}):")
                print(pad_vals.detach().cpu().numpy())
                print(f"Are all padding columns exactly zero? {torch.all(pad_vals == 0.0).item()}")
            print("-------------------------------------------\n")
        
        
        self.last_attn = attn.detach()
        
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = self.proj(x)
        
        return residual + self.dropout(x)
class ConformerConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dropout=0.3, debug=False):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)

        # Pointwise conv (expand channels)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=1
        )

        # Depthwise conv (temporal modeling)
        self.depthwise_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=kernel_size // 2
        )

        # Second normalization
        self.batch_norm = nn.BatchNorm1d(dim)

        # Pointwise projection back
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        """
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]

        # All Conv/Norm operations happen in [B, D, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x) 
        x = self.pointwise_conv2(x)

        x = x.transpose(1, 2) # Back to [B, T, D] once at the end
        x = self.dropout(x)
        return residual + x

class FeedForwardBlock(nn.Module):
    def __init__(self, dim=128, dropout=0.3, debug=False):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim)
        )

        self.dropout = nn.Dropout(dropout)  
        self.debug = debug
    def forward(self, x, mask=None):
        residual = x

        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)

        return residual + 0.5 * x

class ConformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.3, debug=False):
        super().__init__()

        self.attn = AttentionBlock(dim, num_heads, dropout=dropout, debug=debug)
        self.conv = ConformerConvBlock(dim, dropout=dropout, debug=debug)
        self.ffn1 = FeedForwardBlock(dim, dropout=dropout, debug=debug)
        self.ffn2 = FeedForwardBlock(dim, dropout=dropout, debug=debug)
        self.final_norm = nn.LayerNorm(dim) 
        self.debug = debug
        self.last_attn = None

    def forward(self, x, mask=None):
        assert x.dim() == 3, f"Expected [B,T,D], got {x.shape}"
        B, T, D = x.shape
        assert D == x.shape[-1], f"Expected feature dim {x.shape[-1]}, but got {D}"

        x = self.ffn1(x)

        if self.debug:
            print("after FFN:", x.shape)
            print("after FFN mean:", x.mean().item())
            print("after FFN std:", x.std().item())

        assert x.shape == (B, T, D)
        assert not torch.isnan(x).any()

        x = self.attn(x, mask=mask)
        self.last_attn = self.attn.last_attn

        if self.debug:
            print("after attention:", x.shape)
            print("after attention mean:", x.mean().item())
            print("after attention std:", x.std().item())

        # Final safety checks
        assert x.shape == (B, T, D), f"Output shape mismatch: {x.shape}"
        assert not torch.isnan(x).any(), "NaNs detected in attention output"

        x = self.conv(x)
        if self.debug:
            print("after conv:", x.shape)
            print("conv mean:", x.mean().item())
            print("conv std:", x.std().item())

        assert x.shape == (B, T, D), f"Output shape mismatch: {x.shape}"
        assert x.shape[-1] == D, f"Feature dim mismatch: {x.shape}"
        assert not torch.isnan(x).any(), "NaNs detected after Conv"


        x = self.ffn2(x)
        x = self.final_norm(x)

        if self.debug:
            print("after FFN:", x.shape)
            print("after FFN mean:", x.mean().item())
            print("after FFN std:", x.std().item())

        assert x.shape == (B, T, D)
        assert not torch.isnan(x).any()

        return x

class ConformerEncoder(nn.Module):
    def __init__(self, dim=128, num_heads=4, num_layers=3, dropout=0.3, debug=False):
        super().__init__()

        self.debug = debug

        self.dropout = dropout
        self.dim = dim

        self.layers = nn.ModuleList([
            ConformerBlock(dim, num_heads, dropout=dropout, debug=debug)
            for _ in range(num_layers)
        ])

        self.last_attn = []

    def forward(self, x, mask=None):
        assert x.dim() == 3, f"Expected [B,T,D], got {x.shape}"
        B, T, D = x.shape
        assert D == x.shape[-1], f"Expected feature dim {x.shape[-1]}, but got {D}"
        
        self.last_attn = []  # Clear previous attention maps

        for layer in self.layers:
            x = layer(x, mask=mask)
            self.last_attn.append(layer.last_attn)
       
        if self.debug:
            print("Projection dim:", self.dim)
            print("dropout: ", self.dropout)
            print("after encoder:", x.shape)
            print("after encoder mean:", x.mean().item())
            print("after encoder std:", x.std().item())

        assert x.shape == (B, T, D)
        assert not torch.isnan(x).any()
        return x