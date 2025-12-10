"""
Utility functions for converting pretrained weights between different MiT versions.

The original SegFormer/ColonFormer uses a custom MixVisionTransformer with keys like:
- patch_embed1.proj.weight, block1.0.norm1.weight, norm1.weight

But mmseg 2.x uses a different structure:
- layers.0.0.projection.weight, layers.0.1.0.norm1.weight, layers.0.2.weight
"""

import torch
import re


def convert_mit_keys(state_dict):
    """
    Convert pretrained MiT weights from old format to new mmseg 2.x format.
    
    Old format (SegFormer original):
        - patch_embed{i}.proj.weight -> layers.{i-1}.0.projection.weight
        - patch_embed{i}.norm.weight -> layers.{i-1}.0.norm.weight
        - block{i}.{j}.norm1.weight -> layers.{i-1}.1.{j}.norm1.weight
        - block{i}.{j}.attn.q.weight -> layers.{i-1}.1.{j}.attn.attn.in_proj_weight (combined)
        - block{i}.{j}.attn.kv.weight -> (combined into in_proj_weight)
        - block{i}.{j}.attn.proj.weight -> layers.{i-1}.1.{j}.attn.attn.out_proj.weight
        - block{i}.{j}.attn.sr.weight -> layers.{i-1}.1.{j}.attn.sr.weight
        - block{i}.{j}.attn.norm.weight -> layers.{i-1}.1.{j}.attn.norm.weight
        - block{i}.{j}.mlp.fc1.weight -> layers.{i-1}.1.{j}.ffn.layers.0.weight
        - block{i}.{j}.mlp.dwconv.dwconv.weight -> layers.{i-1}.1.{j}.ffn.layers.1.weight
        - block{i}.{j}.mlp.fc2.weight -> layers.{i-1}.1.{j}.ffn.layers.4.weight
        - norm{i}.weight -> layers.{i-1}.2.weight
    
    Args:
        state_dict: Original state dict with old key format
        
    Returns:
        new_state_dict: Converted state dict with new key format
    """
    new_state_dict = {}
    
    # Track q and kv weights to combine into in_proj_weight
    qkv_weights = {}  # key: (layer_idx, block_idx) -> {'q': ..., 'kv': ...}
    qkv_biases = {}
    
    for key, value in state_dict.items():
        new_key = None
        
        # Skip head weights (not used in backbone)
        if key.startswith('head.'):
            continue
            
        # patch_embed{i}.proj -> layers.{i-1}.0.projection
        match = re.match(r'patch_embed(\d+)\.proj\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            param = match.group(2)
            new_key = f'layers.{layer_idx}.0.projection.{param}'
            
        # patch_embed{i}.norm -> layers.{i-1}.0.norm
        match = re.match(r'patch_embed(\d+)\.norm\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            param = match.group(2)
            new_key = f'layers.{layer_idx}.0.norm.{param}'
            
        # norm{i} -> layers.{i-1}.2
        match = re.match(r'norm(\d+)\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            param = match.group(2)
            new_key = f'layers.{layer_idx}.2.{param}'
            
        # block{i}.{j}.norm1/norm2 -> layers.{i-1}.1.{j}.norm1/norm2
        match = re.match(r'block(\d+)\.(\d+)\.norm(\d+)\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            norm_idx = match.group(3)
            param = match.group(4)
            new_key = f'layers.{layer_idx}.1.{block_idx}.norm{norm_idx}.{param}'
            
        # block{i}.{j}.attn.q/kv -> combine into in_proj_weight/bias
        match = re.match(r'block(\d+)\.(\d+)\.attn\.(q|kv)\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            qkv_type = match.group(3)  # 'q' or 'kv'
            param = match.group(4)
            
            cache_key = (layer_idx, block_idx)
            if param == 'weight':
                if cache_key not in qkv_weights:
                    qkv_weights[cache_key] = {}
                qkv_weights[cache_key][qkv_type] = value
            else:
                if cache_key not in qkv_biases:
                    qkv_biases[cache_key] = {}
                qkv_biases[cache_key][qkv_type] = value
            continue  # Don't add directly, will combine later
            
        # block{i}.{j}.attn.proj -> layers.{i-1}.1.{j}.attn.attn.out_proj
        match = re.match(r'block(\d+)\.(\d+)\.attn\.proj\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.attn.attn.out_proj.{param}'
            
        # block{i}.{j}.attn.sr -> layers.{i-1}.1.{j}.attn.sr
        match = re.match(r'block(\d+)\.(\d+)\.attn\.sr\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.attn.sr.{param}'
            
        # block{i}.{j}.attn.norm -> layers.{i-1}.1.{j}.attn.norm
        match = re.match(r'block(\d+)\.(\d+)\.attn\.norm\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.attn.norm.{param}'
            
        # block{i}.{j}.mlp.fc1 -> layers.{i-1}.1.{j}.ffn.layers.0
        # Note: Old format uses Linear (2D), new format uses Conv2d (4D)
        match = re.match(r'block(\d+)\.(\d+)\.mlp\.fc1\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.ffn.layers.0.{param}'
            # Reshape Linear weight (out, in) to Conv2d weight (out, in, 1, 1)
            if param == 'weight' and value.dim() == 2:
                value = value.unsqueeze(-1).unsqueeze(-1)
            
        # block{i}.{j}.mlp.dwconv.dwconv -> layers.{i-1}.1.{j}.ffn.layers.1
        match = re.match(r'block(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.ffn.layers.1.{param}'
            
        # block{i}.{j}.mlp.fc2 -> layers.{i-1}.1.{j}.ffn.layers.4
        # Note: Old format uses Linear (2D), new format uses Conv2d (4D)
        match = re.match(r'block(\d+)\.(\d+)\.mlp\.fc2\.(weight|bias)', key)
        if match:
            layer_idx = int(match.group(1)) - 1
            block_idx = int(match.group(2))
            param = match.group(3)
            new_key = f'layers.{layer_idx}.1.{block_idx}.ffn.layers.4.{param}'
            # Reshape Linear weight (out, in) to Conv2d weight (out, in, 1, 1)
            if param == 'weight' and value.dim() == 2:
                value = value.unsqueeze(-1).unsqueeze(-1)
        
        if new_key is not None:
            new_state_dict[new_key] = value
    
    # Combine q and kv weights into in_proj_weight
    # Old format: q.weight (dim, dim), kv.weight (2*dim, dim)
    # New format: in_proj_weight (3*dim, dim) = cat([q, k, v])
    for (layer_idx, block_idx), weights in qkv_weights.items():
        if 'q' in weights and 'kv' in weights:
            q_weight = weights['q']  # (dim, dim)
            kv_weight = weights['kv']  # (2*dim, dim)
            
            # kv contains k and v concatenated
            dim = q_weight.shape[0]
            k_weight = kv_weight[:dim]
            v_weight = kv_weight[dim:]
            
            # Combine into in_proj_weight: [q, k, v]
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            new_key = f'layers.{layer_idx}.1.{block_idx}.attn.attn.in_proj_weight'
            new_state_dict[new_key] = in_proj_weight
            
    for (layer_idx, block_idx), biases in qkv_biases.items():
        if 'q' in biases and 'kv' in biases:
            q_bias = biases['q']
            kv_bias = biases['kv']
            
            dim = q_bias.shape[0]
            k_bias = kv_bias[:dim]
            v_bias = kv_bias[dim:]
            
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            new_key = f'layers.{layer_idx}.1.{block_idx}.attn.attn.in_proj_bias'
            new_state_dict[new_key] = in_proj_bias
    
    return new_state_dict


def load_pretrained_mit(model, pretrained_path, logger=None):
    """
    Load pretrained MiT weights with automatic key conversion.
    
    Args:
        model: The backbone model (MixVisionTransformer)
        pretrained_path: Path to pretrained weights (.pth file)
        logger: Optional logger
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if conversion is needed by looking at key format
    sample_key = list(state_dict.keys())[0]
    if sample_key.startswith('patch_embed') or sample_key.startswith('block'):
        if logger:
            logger.info('Converting pretrained weights from old format to new mmseg format...')
        else:
            print('[Weight Converter] Converting from old MiT format to new mmseg format...')
        state_dict = convert_mit_keys(state_dict)
    
    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if logger:
        if missing:
            logger.warning(f'Missing keys: {missing}')
        if unexpected:
            logger.warning(f'Unexpected keys: {unexpected}')
    else:
        if missing:
            print(f'[Weight Converter] Missing keys ({len(missing)}): {missing[:5]}...')
        if unexpected:
            print(f'[Weight Converter] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
        
    matched = len(state_dict) - len(unexpected)
    print(f'[Weight Converter] Successfully loaded {matched}/{len(state_dict)} weights')
    
    return missing, unexpected
