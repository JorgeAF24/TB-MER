import torch
# Import torch.nn which contains neural network modules and classes for building neural networks
import torch.nn as nn
# Import torch.nn.functional which contains functions for neural network such as: activation, loss, etc. 
import torch.nn.functional as F
from mlb import MLB
from conformer import ConformerEncoder

class EmotionClassifier(nn.Module):
    def __init__(
            self,
            hidden_dim=128, 
            num_classes=7, 
            dropout=0.3,
            clip_dim=768, # CLIP's default embedding size for both text and image is 768
            proj_dim=128,
            fusion_type="mlb",
            temporal_encoder="conformer",
            debug=False
            ):
        """
        Simple feed-forward classifier for emotion recognition
        Args:
            input_dim (int): size of input vector (text+image embeddings)
            hidden_dim (int): hidden layer size
            num_classes (int): number of emotion categories
            dropout (float): dropout rate
            clip_dim (int): dimension of CLIP embeddings (default 768)
            proj_dim (int): dimension to project CLIP embeddings to (default 128)
        """
        # Defines the structure and learnable parameters of the model, and initializes them.
        # Access to the parent class (nn.Module) methods and attributes
        super(EmotionClassifier, self).__init__()
        self.debug = debug
        self.temporal_encoder = temporal_encoder
        self.proj_dim = proj_dim
        self.text_proj = nn.Linear(clip_dim, proj_dim)
        self.image_proj = nn.Linear(clip_dim, proj_dim)
        # MLB fusion layer to combine the projected text and image features into a joint representation
        self.mlb = MLB(input_dim=proj_dim, rank=128, output_dim=proj_dim) # MLB fusion layer
        self.fusion_dropout = nn.Dropout(0.1) # Dropout after fusion for regularization
        self.fusion_norm = nn.LayerNorm(proj_dim) # Layer normalization after fusion
        # Use a self attention mechanism to capture temporal dependencies in the image features if they are provided as a sequence (e.g., video frames)
        if self.temporal_encoder == "conformer":
            self.temporal_encoder = ConformerEncoder(dim=proj_dim, num_heads=2, num_layers=2, dropout=dropout, debug=self.debug) # Temporal encoder for video data (if needed)
        else: 
            self.temporal_encoder = None
        # Network architecture Fully Connected Layers, with one intermediate hidden layer
        self.fc1 = nn.Linear(proj_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fusion_type = fusion_type
        self.concat_proj = nn.Linear(2 * proj_dim, proj_dim)


    def forward(self, text_features, image_features, mask=None):
        """
        Forward pass
        Args:
            text_features (Tensor): shape [batch_size, clip_dim] [B, 768]
            image_features (Tensor): shape [batch_size, clip_dim] [B,T, 768] OR [B,768]
        Returns:
            logits (Tensor): shape [batch_size, num_classes]
        """

        if self.debug:
            print("input text features:", text_features.shape)
            print("input text features mean:", text_features.mean().item())
            print("input text features std:", text_features.std().item())
            print("input image features:", image_features.shape)
            print("input image features mean:", image_features.mean().item())
            print("input image features std:", image_features.std().item())
        
        
        # Project text 
        text_proj = self.text_proj(text_features) 
        assert text_proj.shape[-1] == self.proj_dim, f"text_proj shape mismatch: {text_proj.shape}"
        
        if self.debug:
            print("text features after projection:", text_proj.shape)
            print("text features mean after projection:", text_proj.mean().item())
            print("text features std after projection:", text_proj.std().item())


        if image_features.dim() == 3:
            # Apply the image projection layer to each frame in the temporal dimension T
            # PyTorch automatically applies the linear layer to the last dimension (D) while keeping the batch and temporal dimensions intact
            image_proj = self.image_proj(image_features) 
            assert image_proj.shape[-1] == self.proj_dim, f"image_proj shape mismatch: {image_proj.shape}"

            if self.debug:
                print("image features after projection:", image_proj.shape)
                print("image features mean after projection:", image_proj.mean().item())
                print("image features std after projection:", image_proj.std().item())

            # Unpack the dimensions of image_proj to get batch size (B), temporal dimension (T), and feature dimension (D)
            B,T,D = image_proj.shape
            
            # Expand text across time
            # Use unsqueeze to add a temporal dimension to text_proj on the second dimension [1]
            text_expanded = text_proj.unsqueeze(1) 
            text_expanded = text_expanded.expand(-1, T, -1) 
        
            # Flatten for MLB
            # To apply MLB fusion, we need to flatten the temporal dimension so that we can process each frame independently.
            image_flat = image_proj.reshape(B * T, D) 
            text_flat = text_expanded.reshape(B * T, D) 
        
            #----FUSION SWITCH----#
            if self.fusion_type == "mlb":
                fused_flat = self.mlb(text_flat, image_flat) 
                if self.debug:
                    print("MLB")
            elif self.fusion_type == "concat":
                fused_flat = torch.cat([text_flat, image_flat], dim=-1)  # [B*T, 512]
                fused_flat = self.concat_proj(fused_flat)  
                if self.debug:
                    print("Concat")
            else:
                raise ValueError(f"Invalid fusion type: {self.fusion_type}")
        

            # Restore temporal structure
            fused_branch = fused_flat.reshape(B, T, -1)
            fused_branch = self.fusion_dropout(fused_branch)
            fused = self.fusion_norm(fused_branch) # Layer normalizat   ion after fusion
            if self.debug:
                print("after MLB:", fused.shape)
                print("MLB mean:", fused.mean().item())
                print("MLB std:", fused.std().item())
            if self.temporal_encoder is not None:
                fused = self.temporal_encoder(fused, mask=mask)  # Apply temporal encoding

            if mask is not None:
                mask = mask.unsqueeze(-1)              # [B,T,1]
                fused = fused * mask                  # zero-out padding
                summed = fused.sum(dim=1)             # sum valid
                counts = mask.sum(dim=1).clamp(min=1) # avoid div 0
                if self.debug:
                    print("mask sum per sample:", mask.sum(dim=1))
                fused = summed / counts
            else:
                fused = fused.mean(dim=1) 
        else:
            image_proj = self.image_proj(image_features)

            if self.fusion_type == "mlb":
                fused = self.mlb(text_proj, image_proj)
                if self.debug:
                    print("Fusion: MLB (no temporal)")
            elif self.fusion_type == "concat":
                fused = torch.cat([text_proj, image_proj], dim=-1)
                fused = self.concat_proj(fused)
                if self.debug:
                    print("Fusion: Concat (no temporal)")
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
            fused = self.fusion_dropout(fused)
            fused = self.fusion_norm(fused)
        # Add activation function between layers and apply dropout for regularization
        # Activation Functions needs to be added in forward() method
        # self.fc1(x) applies the first linear transformation W1x=b
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        # self.fc2(x) applies the second linear transformation W2x=b
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    text = torch.randn(batch_size, 768)
    image = torch.randn(batch_size, 768)
    model = EmotionClassifier()
    output = model(text, image)
    print("Output shape:", output.shape)  # [4, 7]

