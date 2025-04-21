

from typing import Optional
from nntile.layer.add import Add
from nntile.layer.rms_norm import RMSNorm
from nntile.layer.t5_attention import T5Attention
from nntile.model.base_model import BaseModel
from nntile.model.t5_config import T5ConfigNNTile
from nntile.tensor import TensorMoments
from nntile.model.t5_ff import T5LayerFF

from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention as T5LayerSelfAttentionTorch, 
    T5Block as T5BlockTorch, 
    T5Stack as T5StackTorch,
    T5LayerCrossAttention as T5LayerCrossAttentionTorch
)

class T5LayerSelfAttention(BaseModel):
    attention: T5Attention
    layer_norm: RMSNorm
    add: Add

    def __init__(self, x: TensorMoments, attention: T5Attention, layer_norm: RMSNorm, add: Add, config: T5ConfigNNTile):
        self.attention = attention
        self.layer_norm = layer_norm
        self.add = add
        
        layers = [layer_norm, attention, add]
        activations = [x] + layer_norm.activations_output + attention.activations_output + add.activations_output
        super().__init__(activations, layers, config)
        
    @classmethod
    def from_torch(
        cls, 
        torch_layer: T5LayerSelfAttentionTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile, 
        next_tag: int
    ):
        layer_norm, next_tag = RMSNorm.from_torch(torch_layer.layer_norm, x, 0, config.layer_norm_epsilon, next_tag, redux=config.redux)
        attention, next_tag = T5Attention.from_torch(
            torch_layer.SelfAttention, 
            layer_norm.activations_output[0], 
            None, 
            config, 
            next_tag
        )
        add, next_tag = Add.generate_simple(x, attention.activations_output[0], next_tag)
        layer = cls(x, attention, layer_norm, add, config)
        return layer, next_tag


class T5LayerCrossAttention(BaseModel):
    cross_attention: T5Attention
    layer_norm: RMSNorm
    add: Add
    
    def __init__(self, x: TensorMoments, encoder_output: TensorMoments, attention: T5Attention, layer_norm: RMSNorm, add: Add, config: T5ConfigNNTile):
        self.attention = attention
        self.layer_norm = layer_norm
        self.add = add
        
        layers = [layer_norm, attention, add]
        activations = [x, encoder_output] + layer_norm.activations_output + attention.activations_output + add.activations_output
        super().__init__(activations, layers, config)
        
    
    @classmethod
    def from_torch(
        cls, 
        torch_layer: T5LayerCrossAttentionTorch, 
        x: TensorMoments, 
        encoder_output: TensorMoments,
        config: T5ConfigNNTile, 
        next_tag: int
    ):
        layer_norm, next_tag = RMSNorm.from_torch(torch_layer.layer_norm, x, 0, config.layer_norm_epsilon, next_tag, redux=config.redux)
        attention, next_tag = T5Attention.from_torch(
            torch_layer.EncDecAttention, 
            layer_norm.activations_output[0], 
            None, 
            config, 
            next_tag,
            encoder_output=encoder_output
        )
        add, next_tag = Add.generate_simple(x, attention.activations_output[0], next_tag)
        layer = cls(x, encoder_output, attention, layer_norm, add, config)
        return layer, next_tag
    
class T5Block(BaseModel):
    is_decoder: bool
    attention: T5LayerSelfAttention 
    cross_attention: Optional[T5LayerCrossAttention]
    feed_forward: T5LayerFF
    
    def __init__(self, x: TensorMoments, attention: T5LayerSelfAttention, feed_forward: T5LayerFF, config: T5ConfigNNTile, cross_attention = None):
        assert not config.is_decoder or cross_attention is not None, "Cross attention must be provided for decoder blocks"
        
        self.is_decoder = config.is_decoder
        self.attention = attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        layers = attention.layers 
        if cross_attention is not None:
            layers.extend(cross_attention.layers)
        layers.extend(feed_forward.layers)

        activations = [x]
        activations.extend(attention.activations[1:])
        if cross_attention is not None:
            activations.extend(cross_attention.activations)
        activations.extend(feed_forward.activations[1:])
        
        super().__init__(activations, layers, config)
    
    @classmethod
    def from_torch(
        cls, 
        torch_block: T5BlockTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile, 
        next_tag: int,
        encoder_output: TensorMoments = None
    ):
        attention, next_tag = T5LayerSelfAttention.from_torch(torch_block.layer[0], x, config, next_tag)
        cross_attention, next_tag = (
            T5LayerCrossAttention.from_torch(torch_block.layer[1], attention.activations[-1], encoder_output, config, next_tag) 
            if config.is_decoder 
            else (None, next_tag)
        )
        ff_layer_torch = torch_block.layer[2] if config.is_decoder else torch_block.layer[1]
        ff_input = cross_attention.activations[-1] if cross_attention is not None else attention.activations[-1]
        feed_forward, next_tag = T5LayerFF.from_torch(ff_layer_torch, ff_input, config, next_tag)
        block = cls(x, attention, feed_forward, config, cross_attention=cross_attention)
        return block, next_tag
    
    
class T5Stack(BaseModel):
    blocks: list[T5Block]
    final_layer_norm: RMSNorm
    
    def __init__(self, x: TensorMoments, blocks: list[T5Block], final_layer_norm: RMSNorm, config: T5ConfigNNTile):
        self.blocks = blocks
        self.final_layer_norm = final_layer_norm

        activations = [x]
        for block in blocks:
            activations.extend(block.activations[1:])
        activations.append(final_layer_norm.activations_output[0])
        layers = []
        for block in blocks:
            layers.extend(block.layers)
        layers.append(final_layer_norm)
        super().__init__(activations, layers, config)
    
    @classmethod
    def from_torch(
        cls, 
        torch_stack: T5StackTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile,
        next_tag: int,
        encoder_output: TensorMoments = None
    ):
        blocks = []
        next_inp = x
        for layer_idx in range(len(torch_stack.block)):
            torch_block = torch_stack.block[layer_idx]
            block, next_tag = T5Block.from_torch(torch_block, next_inp, config, next_tag, encoder_output=encoder_output)
            blocks.append(block)
            next_inp = block.activations[-1]
            
        final_layer_norm, next_tag = RMSNorm.from_torch(torch_stack.final_layer_norm, next_inp, 0, config.layer_norm_epsilon, next_tag, redux=config.redux)
        
        stack = cls(x, blocks, final_layer_norm, config)
        return stack, next_tag
    