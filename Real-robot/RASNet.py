import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def apply_rope(x):
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    theta = 1.0 / (10000 ** (torch.arange(0, C // 2, dtype=dtype, device=device) / (C // 2)))

    pos_h = torch.arange(H, dtype=dtype, device=device).view(1, H, 1, 1)  # (1, H, 1, 1)
    pos_w = torch.arange(W, dtype=dtype, device=device).view(1, 1, W, 1)  # (1, 1, W, 1)

    pos_emb = pos_h * theta.view(1, 1, 1, C // 2) + pos_w * theta.view(1, 1, 1, C // 2)

    x_rot = x.view(B, C // 2, 2, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, C//2, 2)

    cos = torch.cos(pos_emb)  # (1, H, W, C//2)
    sin = torch.sin(pos_emb)  # (1, H, W, C//2)

    x_real = x_rot[..., 0]
    x_imag = x_rot[..., 1]

    x_real_rot = x_real * cos - x_imag * sin
    x_imag_rot = x_real * sin + x_imag * cos

    x_rotated = torch.stack([x_real_rot, x_imag_rot], dim=-1)  # (B, H, W, C//2, 2)
    x_rotated = x_rotated.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)

    return x_rotated


class DenseBlock(nn.Module):
    """Dense Block for feature extraction with growth rate"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class AdaptiveSpikeNeuron(nn.Module):
    """Adaptive Spiking Neuron with learnable decay factor"""
    def __init__(self, channels, threshold=1.0, reset_potential=0.0):
        super().__init__()
        self.threshold = threshold
        self.reset_potential = reset_potential
        
        # Adaptive decay factor computation
        self.decay_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Membrane potential tracking
        self.register_buffer('membrane_potential', torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute adaptive decay factor
        tau = self.decay_conv(x)
        tau = torch.mean(tau, dim=[2, 3], keepdim=True)  # Channel-wise average
        
        # Simplified spiking mechanism for training stability
        # Replace discrete spiking with smooth approximation
        spike_prob = torch.sigmoid((x - self.threshold) * 5.0)  # Smooth approximation
        
        # Apply adaptive temporal decay
        temporal_output = x * torch.exp(-1.0 / (tau + 1e-6))
        
        # Combine spatial and temporal information
        output = spike_prob * temporal_output + (1 - spike_prob) * x
        
        return output


class RASNet_Block(nn.Module):
    """Recursive Adaptive Spiking Network Block"""
    def __init__(self, dim, kernel_size=3, expansion=4, growth_rate=32):
        super().__init__()

        # Dynamic Recursive Computation branch
        self.r = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)

        # Adaptive Decay Mechanism
        self.w_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.u = nn.Parameter(torch.randn(dim))

        # Spiking Dense Feature Extraction branch
        self.dense_block = DenseBlock(dim, growth_rate=growth_rate, num_layers=4)
        dense_out_channels = dim + 4 * growth_rate
        
        # Linear layers for spatial attention
        self.linear_b = nn.Conv2d(dense_out_channels, dim // 4, 1)
        self.linear_d = nn.Conv2d(dense_out_channels, dim // 4, 1)
        
        # Adaptive spike neuron
        self.spike_neuron = AdaptiveSpikeNeuron(dense_out_channels)
        
        # Final projection layers
        self.proj_spike = nn.Conv2d(dim // 4, dim, 1)
        self.proj_q = nn.Conv2d(dim, dim, 1)
        
        # Channel mixing
        hidden_dim = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, hidden=None):
        B, C, H, W = x.shape

        # === Dynamic Recursive Computation Branch ===
        # Adaptive Decay Mechanism
        w = self.w_conv(x)  # (B, C, H, W)
        w = w.view(B, C, -1)  # (B, C, H*W)

        # Apply RoPE
        x_rope = apply_rope(x)
        r = torch.sigmoid(self.r(x))
        k = self.k(x_rope)
        v = self.v(x_rope)

        k = k.view(B, C, -1)  # (B, C, N)
        v = v.view(B, C, -1)

        # Recurrent WKV mechanism
        if hidden is None:
            numerator = torch.zeros_like(k)
            denominator = torch.zeros_like(k)
        else:
            numerator, denominator = hidden

        new_numerator = numerator * torch.exp(-w) + k * v
        new_denominator = denominator * torch.exp(-w) + k

        wkv = (new_numerator + torch.exp(self.u.view(1, -1, 1)) * k * v) / \
              (new_denominator + torch.exp(self.u.view(1, -1, 1)) * k)

        wkv = wkv.view(B, C, H, W)
        recursive_output = r * wkv

        # === Spiking Dense Feature Extraction Branch ===
        # Dense block processing
        dense_features = self.dense_block(x)  # (B, C+4*growth_rate, H, W)
        
        # Apply adaptive spiking neuron
        spiking_features = self.spike_neuron(dense_features)
        
        # Generate spatial attention
        B_k = self.linear_b(spiking_features)  # (B, C//4, H, W)
        D_k = self.linear_d(spiking_features)  # (B, C//4, H, W)
        
        # Reshape for spatial attention computation
        B_k_flat = B_k.view(B, -1, H * W)  # (B, C//4, H*W)
        D_k_flat = D_k.view(B, -1, H * W)  # (B, C//4, H*W)
        
        # Compute spatial attention (simplified version)
        attention = torch.bmm(B_k_flat.transpose(1, 2), B_k_flat)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to D_k
        attended_features = torch.bmm(attention, D_k_flat.transpose(1, 2))  # (B, H*W, C//4)
        attended_features = attended_features.transpose(1, 2).view(B, -1, H, W)  # (B, C//4, H, W)
        
        # Project to match dimensions
        spike_output = self.proj_spike(attended_features)  # (B, C, H, W)
        
        # === Guided Self-Attention Fusion ===
        # Combine recursive and spiking features
        combined_features = recursive_output + spike_output
        
        # Generate Q, K, V for self-attention
        Q = self.proj_q(combined_features)
        K = self.k(combined_features)
        V = self.v(combined_features)
        
        # Self-attention
        Q_flat = Q.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        K_flat = K.view(B, C, -1)  # (B, C, H*W)
        V_flat = V.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        attention_scores = torch.bmm(Q_flat, K_flat) / (C ** 0.5)  # (B, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_output = torch.bmm(attention_weights, V_flat)  # (B, H*W, C)
        attended_output = attended_output.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # Channel mixing
        residual = x
        x = residual + attended_output
        x = x + self.fc2(F.relu(self.fc1(x)) ** 2)

        return x, (new_numerator.detach(), new_denominator.detach())


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(ch, out_channels, 1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        merged = []
        prev = None
        for lateral in reversed(laterals):
            if prev is not None:
                lateral += F.interpolate(prev, scale_factor=2, mode='nearest')
            prev = lateral
            merged.insert(0, lateral)

        return [self.output_convs[i](feat) for i, feat in enumerate(merged)]


class RASNet_ImageModel(nn.Module):
    """Recursive Adaptive Spiking Network for Image Processing"""
    def __init__(self, num_classes=6):
        super().__init__()

        # Stem layer with SReLU activation
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # Using ReLU instead of SReLU for simplicity
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Three hierarchical processing stages
        self.stage1 = nn.ModuleList([RASNet_Block(64) for _ in range(2)])
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.stage2 = nn.ModuleList([RASNet_Block(128) for _ in range(2)])
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.stage3 = nn.ModuleList([RASNet_Block(256) for _ in range(2)])

        # Multi-scale feature fusion with FPN
        self.fpn = FPN([64, 128, 256], 256)
        
        # Learnable fusion weights
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Stem processing
        x = self.stem(x)

        # Stage 1 processing
        hidden1 = [None] * len(self.stage1)
        for i, block in enumerate(self.stage1):
            x, hidden1[i] = block(x)
        f1 = x

        # Stage 2 processing
        x = self.down1(f1)
        hidden2 = [None] * len(self.stage2)
        for i, block in enumerate(self.stage2):
            x, hidden2[i] = block(x)
        f2 = x

        # Stage 3 processing
        x = self.down2(f2)
        hidden3 = [None] * len(self.stage3)
        for i, block in enumerate(self.stage3):
            x, hidden3[i] = block(x)
        f3 = x

        # Multi-scale feature fusion
        features = self.fpn([f1, f2, f3])
        
        # Weighted fusion with learnable parameters
        f1_proj = nn.Conv2d(256, 256, 1).to(x.device)(features[0])
        f2_up = F.interpolate(features[1], scale_factor=2, mode='nearest')
        f3_up = F.interpolate(features[2], scale_factor=4, mode='nearest')
        
        fused = self.alpha1 * f1_proj + self.alpha2 * f2_up + self.alpha3 * f3_up

        return self.head(fused)


# Test the model
if __name__ == "__main__":
    model = RASNet_ImageModel(num_classes=6)
    x = torch.rand(2, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")