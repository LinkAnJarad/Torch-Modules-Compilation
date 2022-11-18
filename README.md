# Torch-Modules-Compilation
A compilation of various torch modules used in ML papers. More to be added!

## Table of Contents
- [Modules](#modulesblocks)
    - [Bottleneck Residual Block](#bottleneck-residual-block)
    - [Depthwise Seperable Convolution](#depthwise-seperable-convolution)
    - [SAGAN self-attention module](#sagan-self-attention-module)
    - [Global-Local Attention Module](#global-local-attention-module)
    - [Global Context Module](#global-context-module)
    - [LFSA Tokenizer and Refinement Block](#lfsa-tokenizer-and-refinement-block)
    - [Parameter-Free Channel Attention (PFCA)](#parameter-free-channel-attention-pfca)
    - [Patch Merger](#patch-merger)
    - [ResBlock](#resblock)
    - [Up/Down sample ResBlock](#updown-sample-resblock)
    - [Residual MLP Block](#residual-mlp-block)
    - [Residual MLP Downsampling Block](#residual-mlp-downsampling-block)
    - [Transformer Encoder Layer](#transformer-encoder-layer)
    - [UNet Encoder and Decoder](#unet-encoder-and-decoder)

- [License](#license)

# Modules/Blocks

## Bottleneck Residual Block
![image](https://user-images.githubusercontent.com/79294502/202608237-bf9bf8c8-a409-4157-ae69-75dc25896e6a.png)

Your basic bottleneck residual block in ResNets.
Image from the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf)

### Parameters
`in_channels` (int): number of input channels

`bottleneck_channels` (int): number of bottleneck channels; usually less than the number of bottleneck channels

`dropout` (float): dropout rate; performed after every convolution
    
### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 256, 16, 16) # (batch_size, channels, height, width)
block = modules.BottleneckResBlock(in_channels=256, bottleneck_channels=64)

block(x).shape # (32, 256, 16, 16)
```

## Depthwise Seperable Convolution

![image](https://user-images.githubusercontent.com/79294502/202608395-fe1aabc6-1aac-473a-a734-ab4c9527b81a.png)

A depthwise seperable convolution; consists of a depthwise convolution and a pointwise convolution. Used in MobileNets and used in the paper ["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/pdf/1704.04861v1.pdf). Image also from this paper.

### Parameters: 

`in_channels` (int): Number of input channels

`out_channels` (int): Number of output channels

`kernel_size` (int): Size of depthwise convolution kernel

`stride` (int): Stride of depthwise convolution

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)
block = modules.DepthwiseSepConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

block(x).shape # (32, 128, 16, 16)
```

## SAGAN self-attention module
![image](https://user-images.githubusercontent.com/79294502/202611126-ed4b6a88-5a7f-4f47-b89c-e4b9188b4db7.png)

A feature map self-attention module used in SAGAN; ["Self-Attention Generative Adversarial Networks"](https://arxiv.org/pdf/1805.08318.pdf). Image also from this paper. This code implementation was copied and modified from https://github.com/rosinality/sagan-pytorch/blob/master/model.py#L82 under Apache 2.0 License. Modification removes spectral initalization.
    
### Parameters
`in_channels` (int): Number of input channels

### Usage
```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)
block = modules.FeatureMapSelfAttention(in_channels=64)

block(x).shape # (32, 64, 16, 16)
```

## Global-Local Attention Module
![image](https://user-images.githubusercontent.com/79294502/202611948-fe8a9eb4-e0b4-4440-8710-d386d4ebdeb2.png)


An convolutional attention module introduced in the paper ["All the attention you need: Global-local, spatial-channel attention for image retrieval."](https://openaccess.thecvf.com/content/WACV2022/papers/Song_All_the_Attention_You_Need_Global-Local_Spatial-Channel_Attention_for_Image_WACV_2022_paper.pdf). Image also from this paper.
        
### Parameters
`in_channels` (int): number of channels of the input feature map

`num_reduced_channels` (int): number of channels that the local and global spatial attention modules will reduce the input feature map. Refer to figures 3 and 5 in the paper.

`feaure_map_size` (int): height/width of the feature map

`kernel_size` (int): scope of the inter-channel attention

### Usage
```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)

block = modules.GLAM(in_channels=64, num_reduced_channels=48, feature_map_size=16, kernel_size=5)
# height and width is equal to feature_map_size

block(x).shape # (32, 64, 16, 16)
```

## Global Context Module
![image](https://user-images.githubusercontent.com/79294502/202612104-8613e1bb-c3b9-4ad2-a66d-ec29937afa1a.png)

A sort of self-attention (non-local) block on feature maps. Implementation of ["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/pdf/1904.11492.pdf).
    
### Parameters

`input_channels` (int): Number of input channels

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)

block = modules.GlobalContextModule(input_channels=64)

block(x).shape # (32, 64, 16, 16)
```

## LFSA Tokenizer and Refinement Block
![image](https://user-images.githubusercontent.com/79294502/202612366-b37dc304-34e5-4bfb-96ad-2c54ee57bc8c.png)

Implementation of the tokenizer in ["Learning Token-Based Representation for Image Retrieval"](https://arxiv.org/pdf/2112.06159.pdf) This are two modules: The tokenizer module that converts feature maps from a CNN (in the paper's case, feature maps from a local-feature-self-attention module) and tokenizes them "into L visual tokens". This is used prior to the refinement block as described in the paper. The refinement block "enhance[s] the obtained visual tokens with self-attention and cross-attention."

### Parameters

**LFSA Tokenizer**

`in_channels` (int): number of input channels

`num_att_maps` (int): number of tokens to tokenize the input into; also the number of channels used by the spatial attention

**Refinement Block**

`d_model` (int): dimensionality/channels of input

`nhead` (int): number of attention heads in the transformer

`dim_feedforward` (int): number of hidden dimensions in the feedforward layers

`dropout` (int): dropout rate

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)

tokenizer = modules.LFSATokenizer(in_channels=64, num_att_maps=48)
refinement_block = modules.RefinementBlock(d_model=64, nhead=2, dim_feedforward=48*4, dropout=0.1)

visual_tokens, cnn_output = tokenizer(x)
print(visual_tokens.shape) # (32, 48, 64)
print(cnn_output.shape) # (32, 16*16, 64)

output = refinement_block(visual_tokens, cnn_output)
print(output.shape) # (32, 48, 64)
```

## Parameter-Free Channel Attention (PFCA)
![image](https://user-images.githubusercontent.com/79294502/202614077-9a337542-dd57-4bce-b278-30e2108f59b7.png)

A channel attention module for convolutional feature maps without any trainable parameters. Used in and image from the paper ["PARAMETER-FREE CHANNEL ATTENTION FOR IMAGE CLASSIFICATION AND SUPER-RESOLUTION"](https://www.researchgate.net/publication/360462671_PARAMETER-FREE_CHANNEL_ATTENTION_FOR_IMAGE_CLASSIFICATION_AND_SUPER-RESOLUTION).

### Parameters

`feature_map_size` (int): Length/width of the input feature map

`_lambda` (float): A hyperparameter that is added to the variance (default: 1e-4)

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, channels, height, width)
block = modules.ParameterFreeChannelAttention(feature_map_size=16)

block(x).shape # (32, 64, 16, 16)
```

## Patch Merger
![image](https://user-images.githubusercontent.com/79294502/202614966-bdca4891-3987-466e-8f5f-4f8d3343520d.png)

Merges N tokens into M tokens in transformer models. Typically added in-between transformer layers. Introduced in the paper ["LEARNING TO MERGE TOKENS IN VISION TRANSFORMERS"](https://arxiv.org/pdf/2202.12015v1.pdf). Image from this paper. Copied from [lucidrains' repo](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_with_patch_merger.py) under the MIT license.

### Parameters

`dim` (int): dimensionality/channels of the tokens

`output_tokens` (int): number of output merged tokens

`norm` (bool): normalize the input before merging

`scale` (bool): scale the attention matrix by the square root of dim (for numerical stability)

### Usage
```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16) # (batch_size, seq_length, channels)
block = modules.PatchMerger(dim=16, output_tokens=48, scale=True)

block(x).shape # (32, 48, 16)
```

## ResBlock
![image](https://user-images.githubusercontent.com/79294502/202616071-ce43efbf-433f-414f-adcc-142ea4ae78f8.png)

Your basic residual block. Used in ResNets. Image from original paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf)
    
### Parameters

`in_channels` (int): number of input channels

`kernel_size` (int): kernel size

`dropout` (float): dropout rate

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 16, 16) # (batch_size, seq_length, channels)
block = modules.ResBlock(in_channels=64, kernel_size=3, dropout=0.2)

block(x).shape # (32, 64, 16, 16)
```

## Up/Down sample ResBlock
Composed of several residual blocks and a down/up sampling at the end; adapted from [Stable Diffusion's ResnetBlock](https://github.com/CompVis/stable-diffusion/blob/ce05de28194041e030ccfc70c635fe3707cdfc30/ldm/modules/diffusionmodules/model.py#L82).

### Parameters

`in_channels` (int): number of input channels

`out_channels` (int): number of output channels

`num_groups` (int): number of groups for Group Normalization

`num_layers` (int): number of residual blocks

`dropout` (float): dropout rate

`sample` (str): One of "down", "up", or "none". For downsampling 2x, use "down". For upsampling 2x, use "up". Use "none" for no down/up sampling.

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 64, 96, 96) # (batch_size, seq_length, channels)
block = modules.ResBlockUpDownSample(
    in_channels=64, 
    out_channels=128, 
    num_groups=8, 
    num_layers=2, 
    dropout=0.1, 
    sample='down'
)

block(x).shape # (32, 128, 48, 48)
```

## Residual MLP Block

An improvement of standard MLPs along with residual connections. From ["Generalizing MLPs With Dropouts, Batch Normalization, and Skip Connections"](https://arxiv.org/pdf/2108.08186.pdf). This implements the residual MLP block (eq. 5 in the paper).
    
### Parameters

`dim` (int): number of input dimensions

`ic_first` (bool): normalize and dropout at the start

`dropout` (float): dropout rate

### Usage

```python
from torch_modules_compilation import modules

x = torch.randn(32, 96) # (batch_size, dim)
block = modules.ResidualMLP_block(dim=96, ic_first=True, dropout=0.1)

block(x).shape # (32, 96)
```

## Residual MLP Downsampling Block

An improvement of standard MLPs along with residual connections. From ["Generalizing MLPs With Dropouts, Batch Normalization, and Skip Connections"](https://arxiv.org/pdf/2108.08186.pdf). This implements the residual MLP block (eq. 5 in the paper).

### Parameters
`dim` (int): number of input dimensions

`downsample_dim` (int): number of output dimensions

`dropout` (float): dropout rate

### Usage
```python
from torch_modules_compilation import modules

x = torch.randn(32, 96) # (batch_size, dim)
block = modules.ResidualMLP_downsample(dim=96, downsample_dim=48, dropout=0.1)

block(x).shape # (32, 48)
```

## Transformer Encoder Layer
Standard transformer encoder layer with queries, keys, and values as inputs.
    
### Parameters

`d_model` (int): model dimensionality

`nhead` (int): number of attention heads

`dim_feedforward` (int): number of hidden dimensions in the feedforward layers

`dropout` (float): dropout rate

`kdim` (int, optional): dimensions of the keys

`vdim` (int, optional): dimensions of the values

### Usage

```python
from torch_modules_compilation import modules

queries = torch.randn(32, 20, 64) # (batch_size, seq_length, dim)
keys = torch.randn(32, 19, 48) # (batch_size, seq_length, dim)
values = torch.randn(32, 19, 96) # (batch_size, seq_length, dim)

block = modules.TransformerEncoderLayer(
    d_model=64,
    nhead=8, 
    dim_feedforward=256,
    dropout=0.2,
    kdim=48,
    vdim=96
)

block(queries, keys, values).shape # (32, 20, 64)
```

## UNet Encoder and Decoder
![image](https://user-images.githubusercontent.com/79294502/202618135-a0b6e0f1-db4e-433e-bbaa-a1c5d104215d.png)

Standard UNet implementation. From the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf).

### Parameters

**UNet Encoder**

`channels` (list of ints): A list containing the number of channels in the encoder. E.g [3, 64, 128, 256]

`dropout` (float): dropout rate

**UNet Decoder**

`channels` (list of ints): A list containing the number of channels in the encoder. E.g. [256, 128, 64, 3]

`dropout` (float): dropout rate

### Usage

```python
from torch_modules_compilation import modules

images = torch.randn(16, 3, 224, 224) # (batch_size, channels, height, width)

unet_encoder = modules.UnetEncoder(channels=[3,64,128,256], dropout=0.1)
unet_decoder = modules.UnetDecoder(channels=[256,128,64,3], dropout=0.1)

encoder_features = unet_encoder(images)

output = unet_decoder(encoder_features)
print(output.shape) # (16, 64, 224, 224)
```

# License
Unless specified, some of these modules are licensed under various licenses and/or copied from other repositories, such as MIT and Apache. Take note of these licenses when using these code in your work. The rest are of my own implementation, which is under the MIT license.
