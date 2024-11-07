
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from enum import Enum, auto
from networks import basic_avatar, volumetric_avatar
from aitypes import *


import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from networks.volumetric_avatar.utils import replace_bn_to_bcn,replace_bn_to_gn,replace_bn_to_in,ResBlock,norm_layers,activations


from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models




@dataclass
class IdtEmbedConfig:
    """Configuration for Identity Embedding Network.

    Attributes:
        idt_backbone: Name of the backbone architecture (e.g., 'resnet18')
        num_source_frames: Number of source images per identity
        idt_output_size: Size of the output feature map
        idt_output_channels: Number of output channels
        num_gpus: Number of GPUs to use
        norm_layer_type: Type of normalization layer ('bn', 'in', 'gn', 'bcn')
        idt_image_size: Input image size
    """
    idt_backbone: str ="resnet50"
    num_source_frames: int=1
    idt_output_size: int=8
    idt_output_channels: int=512
    num_gpus: int=1
    norm_layer_type: str="gn"
    idt_image_size: int=256


class IdtEmbed(nn.Module):
    """Identity Embedding Network.
    
    This module processes source images to create identity embeddings using a backbone
    CNN architecture with customizable normalization layers.
    """

    def __init__(self, cfg: IdtEmbedConfig):
        """Initialize the Identity Embedding network.

        Args:
            cfg: Configuration object containing network parameters
        """
        super().__init__()
        self.cfg = cfg
        self._setup_network()
        self._register_normalization_values()

    def _setup_network(self) -> None:
        """Set up the network architecture including backbone and custom layers."""
        expansion = self._get_expansion_factor()
        self.net = self._create_backbone()
        self.net.avgpool = nn.AdaptiveAvgPool2d(self.cfg.idt_output_size)
        self.net.fc = nn.Conv2d(
            in_channels=512 * expansion,
            out_channels=self.cfg.idt_output_channels,
            kernel_size=1,
            bias=False
        )
        self._setup_normalization_layers()

    def _get_expansion_factor(self) -> int:
        """Determine the expansion factor based on backbone architecture."""
        return 1 if self.cfg.idt_backbone == 'resnet18' else 4

    def _create_backbone(self) -> nn.Module:
        """Create the backbone network."""
        return getattr(models, self.cfg.idt_backbone)(pretrained=True)

    def _setup_normalization_layers(self) -> None:
        """Set up normalization layers based on configuration."""
        norm_types = {
            'in': (replace_bn_to_in, 'Instance Normalization'),
            'gn': (replace_bn_to_gn, 'Group Normalization'),
            'bcn': (replace_bn_to_bcn, 'Batch-Channel Normalization'),
        }

        if self.cfg.norm_layer_type == 'bn':
            return
        
        if self.cfg.norm_layer_type not in norm_types:
            raise ValueError(f"Unsupported normalization type: {self.cfg.norm_layer_type}")
        
        replace_fn, norm_name = norm_types[self.cfg.norm_layer_type]
        self.net = replace_fn(self.net, 'IdtEmbed')

    def _register_normalization_values(self) -> None:
        """Register normalization values as buffers."""
        self.register_buffer('mean', 
            torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', 
            torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after passing through all network layers
        """
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.fc(x)
        x = self.net.avgpool(x)
        return x

    def forward_image(self, source_img: torch.Tensor) -> torch.Tensor:
        """Process source images to create identity embeddings.

        Args:
            source_img: Input source images tensor of shape (B*N, C, H, W)
                where N is num_source_frames

        Returns:
            Identity embedding tensor
        """
        source_img = F.interpolate(
            source_img, 
            size=(self.cfg.idt_image_size, self.cfg.idt_image_size), 
            mode='bilinear'
        )
        n = self.cfg.num_source_frames
        b = source_img.shape[0] // n

        inputs = (source_img - self.mean) / self.std
        idt_embed_tensor = self._forward_impl(inputs)
        return idt_embed_tensor.view(b, n, *idt_embed_tensor.shape[1:]).mean(1)

    def forward(self, source_img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            source_img: Input source images tensor

        Returns:
            Identity embedding tensor
        """
        return self.forward_image(source_img)


@dataclass(frozen=True)
class LocalEncoderConfig:
    """Hardcoded configuration for LocalEncoder."""
    # Generator parameters
    gen_upsampling_type: str = "trilinear"
    gen_downsampling_type: str = "avgpool"
    gen_input_image_size: int = 512
    gen_latent_texture_size: int = 64
    gen_latent_texture_depth: int = 16
    gen_latent_texture_channels: int = 96
    gen_num_channels: int = 32
    gen_max_channels: int = 512
    gen_activation_type: str = "relu"
    
    # Encoder parameters
    enc_channel_mult: float = 4.0
    enc_block_type: str = "res"
    
    # Normalization and other parameters
    norm_layer_type: str = "gn"
    num_gpus: int = 8
    warp_norm_grad: bool = False
    in_channels: int = 3


class LocalEncoder(nn.Module):
    """
    LocalEncoder with hardcoded configuration for processing input images into latent representations.
    """
    def __init__(self):
        super().__init__()
        self.cfg = LocalEncoderConfig()
        self._init_parameters()
        self._build_network()

    def _init_parameters(self):
        """Initialize model parameters and computed values."""
        self.ratio = self.cfg.gen_input_image_size // self.cfg.gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))
        self.init_depth = self.cfg.gen_latent_texture_depth
        self.spatial_size = self.cfg.gen_input_image_size
        
        # Determine normalization type
        self.norm_type = (
            self.cfg.norm_layer_type if self.cfg.norm_layer_type != 'bn'
            else 'sync_bn' if self.cfg.num_gpus >= 2 else 'bn'
        )

    def _build_network(self):
        """Construct the network architecture."""
        # Initialize grid sample if needed
        if self.cfg.warp_norm_grad:
            from . import GridSample
            self.grid_sample = GridSample(self.cfg.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(
                inputs.float(), 
                grid.float(), 
                padding_mode='reflection'
            )

        # Initial convolution
        out_channels = int(self.cfg.gen_num_channels * self.cfg.enc_channel_mult)
        self.initial_conv = nn.Conv2d(
            in_channels=self.cfg.in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=3
        )

        # Build downsampling blocks
        self.down_blocks = nn.ModuleList()
        spatial_size = self.spatial_size
        
        for i in range(self.num_2d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.cfg.gen_max_channels)
            
            block = self._create_down_block(
                in_channels, 
                out_channels, 
                spatial_size
            )
            self.down_blocks.append(block)
            spatial_size //= 2

        # Final processing layers
        self.finale = self._create_finale_layers(out_channels)

    def _create_down_block(self, in_channels, out_channels, spatial_size):
        """Create a single downsampling block."""

        return ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            norm_layer_type=self.norm_type,
            activation_type=self.cfg.gen_activation_type,
            resize_layer_type=self.cfg.gen_downsampling_type
        )

    def _create_finale_layers(self, in_channels):
        """Create the final processing layers."""

        layers = []
        
        # Add normalization and activation for residual blocks
        if self.cfg.enc_block_type == 'res':
            layers.extend([
                norm_layers[self.norm_type](in_channels),
                activations[self.cfg.gen_activation_type](inplace=True)
            ])

        # Add final convolution
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.cfg.gen_latent_texture_channels * self.init_depth,
                kernel_size=1
            )
        )
        
        return nn.Sequential(*layers)

    def forward(self, source_img):
        """
        Forward pass of the encoder.
        
        Args:
            source_img: Input image tensor
            
        Returns:
            Encoded representation of the input image
        """
        x = self.initial_conv(source_img)
        
        # Apply downsampling blocks
        for block in self.down_blocks:
            x = block(x)
            
        # Apply final processing
        x = self.finale(x)
        
        return x


# Example usage:
def create_encoder():
    """Create an instance of the LocalEncoder with hardcoded config."""
    return LocalEncoder()


# Optional utility function to get the config without creating the model
def get_encoder_config():
    """Get the hardcoded configuration used by LocalEncoder."""
    return LocalEncoderConfig()


@dataclass
class ModelConfig:
    """Main configuration class for the volumetric avatar model"""
    
    # Basic model parameters
    image_size: int = 256
    latent_dim: int = 512
    num_source_frames: int = 1
    batch_size: int = 8
    
    # Network architecture
    activation_type: ActivationType = ActivationType.RELU
    norm_layer_type: NormLayerType = NormLayerType.BATCH_NORM
    use_spectral_norm: bool = True
    use_weight_standardization: bool = False
    
    # Encoder settings
    encoder_channels: int = 32
    encoder_max_channels: int = 512
    encoder_channel_multiplier: float = 2.0
    encoder_block_type: BlockType = BlockType.RESIDUAL
    
    # Volume settings
    volume_channels: int = 64
    volume_size: int = 64
    volume_depth: int = 16
    volume_blocks: int = 4
    use_volume_renderer: bool = False
    
    # Decoder settings
    decoder_channels: int = 32
    decoder_max_channels: int = 512
    decoder_blocks: int = 8
    decoder_channel_multiplier: float = 2.0
    decoder_predict_segmentation: bool = False
    
    # Training settings
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    adversarial_loss_weight: float = 1.0
    feature_matching_weight: float = 60.0
    vgg_loss_weight: float = 20.0
    
    # Discriminator settings
    discriminator_channels: int = 64
    discriminator_max_channels: int = 512
    discriminator_blocks: int = 4
    discriminator_scales: int = 2
    use_style_discriminator: bool = False
    
    # Advanced features
    use_adaptive_convolution: bool = False
    use_adaptive_normalization: bool = False
    use_mixing_generator: bool = True
    predict_cycle: bool = True
    
    # Volume renderer settings (if enabled)
    @dataclass
    class VolumeRendererConfig:
        depth_resolution: int = 48
        hidden_dim: int = 448
        num_layers: int = 2
        squeeze_dim: int = 0
        features_sigmoid: bool = True
    
    volume_renderer_config: VolumeRendererConfig = field(default_factory=VolumeRendererConfig)
    
    # Expression embedding settings
    @dataclass
    class ExpressionConfig:
        backbone: str = "resnet18"
        output_channels: int = 512
        output_size: int = 4
        dropout: float = 0.0
        use_smart_scaling: bool = False
        max_scale: float = 0.75
        max_angle_tolerance: float = 0.8
    
    expression_config: ExpressionConfig = field(default_factory=ExpressionConfig)
    
    # Identity embedding settings
    @dataclass
    class IdentityConfig:
        backbone: str = "resnet50"
        output_channels: int = 512
        output_size: int = 4
        image_size: int = 256
    
    identity_config: IdentityConfig = field(default_factory=IdentityConfig)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.num_source_frames != 1:
            raise ValueError("Multiple source frames are not supported")
        
        if self.volume_size % 2 != 0:
            raise ValueError("Volume size must be even")
            
        if self.volume_depth % 2 != 0:
            raise ValueError("Volume depth must be even")
    
    @property
    def device_count(self) -> int:
        """Get number of available GPU devices"""
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def get_optimizer_config(self, optimizer_type: str = "adam") -> Dict:
        """Get optimizer configuration"""
        return {
            "lr": self.learning_rate,
            "betas": (self.beta1, self.beta2),
            "eps": 1e-8
        }

    def get_scheduler_config(self, scheduler_type: str = "cosine") -> Dict:
        """Get learning rate scheduler configuration"""
        return {
            "T_max": 250000,
            "eta_min": 1e-6
        }
    
@dataclass
class NetworkConfig:
    """Configuration for network architectures"""
    local_encoder: Dict = None
    volume_renderer: Dict = None
    idt_embedder: Dict = None 
    exp_embedder: Dict = None
    warp_generator: Dict = None
    decoder: Dict = None
    discriminator: Dict = None
    unet3d: Dict = None
    
class Model(nn.Module):
    """Main model class for volumetric avatar generation"""

    def __init__(self, config: ModelConfig, training: bool = True, rank: int = 0, exp_dir: Optional[str] = None):
        super().__init__()
        self.config = config
        self.exp_dir = exp_dir
        self.rank = rank
        
        # Basic parameters
        self.num_source_frames = 1  # No support for multiple sources
        self.background_net_input_channels = 64
        self.embed_size = 8
        
        # Model settings
        self.pred_seg = False
        self.use_stylegan_d = False
        self.pred_flip = False
        self.pred_mixing = True
        self.pred_cycle = True
        
        # Initialize components
        self.setup_preprocessing()
        self.init_networks(training)
        if training:
            self.init_losses()
            
        # Register coordinate grids
        self.register_coordinate_grids()
        
        # Initialize training state
        self.prev_targets = None
        self.thetas_pool = []
        
        # Apply initializations
        self.apply(weight_init.weight_init(init_type='kaiming', init_gain=0.02))
        
    def setup_preprocessing(self):
        """Setup preprocessing functions"""
        self.resize_d = lambda img: F.interpolate(
            img, 
            mode='bilinear',
            size=(224, 224), 
            align_corners=False
        )
        
        self.resize_u = lambda img: F.interpolate(
            img,
            mode='bilinear', 
            size=(256, 256),
            align_corners=False
        )
        
    def register_coordinate_grids(self):
        """Register 2D and 3D coordinate grids as buffers"""
        # 2D identity grid
        grid_s = torch.linspace(-1, 1, self.config.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer(
            'identity_grid_2d',
            torch.stack([u, v], dim=2).view(1, -1, 2),
            persistent=False
        )

        # 3D identity grid
        grid_s = torch.linspace(-1, 1, self.config.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.config.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer(
            'identity_grid_3d',
            torch.stack([u, v, w, e], dim=3).view(1, -1, 4),
            persistent=False
        )

    def init_networks(self, training: bool = True):
        """Initialize all network components"""
        # Encoders

        self.local_encoder = LocalEncoder()

        # Initialize face parsing if needed
        if self.config.use_mix_mask:
            self.face_parsing = volumetric_avatar.FaceParsing()
            
        @dataclass
        class Config:
            z_dim: int = 16,  # Input latent (Z) dimensionality.
            c_dim: int = 96,  # Conditioning label (C) dimensionality.
            w_dim: int = 64,  # Intermediate latent (W) dimensionality.
            img_resolution: int = 64,  # Output resolution.
            dec_channels: int = 1024,  # Number of output color channels
            img_channels: int = 384,  # Number of output color channels
            features_sigm: int  = 1,
            squeeze_dim: int = 0,
            depth_resolution: int = 48,
            hidden_vol_dec_dim: int = 448,
            num_layers_vol_dec: int  = 2

        cfg = Config()
        # Volume renderer
        self.volume_renderer = volumetric_avatar.VolumeRenderer(
            cfg
        )

            
        # Identity and Expression Embedders
        idtCfg = IdtEmbedConfig()
        self.idt_embedder = IdtEmbed(idtCfg)

        # Expression Embedder 
        self.expression_embedder = volumetric_avatar.ExpressionEmbed(
            use_amp_autocast=False,
            lpe_head_backbone=self.config.lpe_face_backbone,
            lpe_face_backbone="resnet18",
            image_size=256,
            num_gpus=self.config.num_gpus,
            lpe_output_channels=128,
            lpe_final_pooling_type="avg",
            lpe_output_size=4,
            norm_layer_type=self.config.norm_layer_type
        )

        # Warp generators
        warp_gen_config = {
            "eps": 1e-8,
            "num_gpus": self.config.num_gpus,
            "use_amp_autocast": False,
            "gen_adaptive_conv_type": "sum",
            "gen_activation_type": "relu", 
            "gen_upsampling_type": "trilinear",
            "gen_downsampling_type": "avgpool",
            "gen_max_channels": self.config.dec_max_channels,
            "gen_num_channels": 32,
            "warp_channel_mult": 1.0,
            "warp_block_type": "res",
            "norm_layer_type": self.config.norm_layer_type
        }
        
        self.xy_generator = volumetric_avatar.WarpGenerator(
            **warp_gen_config, 
            input_channels=self.config.dec_max_channels
        )
        
        self.uv_generator = volumetric_avatar.WarpGenerator(
            **warp_gen_config,
            input_channels=self.config.dec_max_channels 
        )

        # Initialize discriminators for training
        if training:
            self.init_discriminators()

    def init_discriminators(self):
        """Initialize discriminator networks"""
        self.discriminator = basic_avatar.MultiScaleDiscriminator(
            min_channels=64,
            max_channels=512, 
            num_blocks=4,
            input_channels=3,
            input_size=self.config.image_size,
            num_scales=2
        )
        
        if self.config.use_stylegan_d:
            self.stylegan_discriminator = basic_avatar.DiscriminatorStyleGAN2(
                size=self.config.image_size,
                channel_multiplier=1,
                my_ch=2
            )

    def init_losses(self):
        """Initialize loss functions"""
        if self.config.pred_seg:
            self.seg_loss = nn.BCELoss()
            
        self.init_additional_losses()

 