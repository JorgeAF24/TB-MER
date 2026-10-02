import torch
import torch.nn as nn
import torch.nn.functional as F


class MLB(nn.Module):
    def __init__(self, input_dim=256, rank=128, output_dim=128):
        """
        Low-rank bilinear pooling (MLB)

        Args:
            input_dim: dimension of each modality (after projection)
            rank: low-rank dimension (r)
            output_dim: final fused dimension (d)
        """
        super(MLB, self).__init__()

        # Uᵀ x
        self.U = nn.Linear(input_dim, rank)

        # Vᵀ y
        self.V = nn.Linear(input_dim, rank)

        # Pᵀ (projection back)
        self.P = nn.Linear(rank, output_dim)

    def forward(self, x, y):
        """
        Args:
            x: [B,input_dim]
            y: [B,input_dim]

        Returns:
            z: [B,output_dim]
        """

        x_proj = torch.tanh(self.U(x))   # [B,r]
        y_proj = torch.tanh(self.V(y))   # [B,r]

        # Hadamard product (element-wise multiplication)
        joint = x_proj * y_proj          # [B,r]

        z = self.P(joint)                # [B,output_dim]

        return z