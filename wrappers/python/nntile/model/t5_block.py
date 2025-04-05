

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
    T5Stack as T5StackTorch
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

    
class T5Block(BaseModel):
    is_decoder: bool
    attention: T5LayerSelfAttention 
    # cross_attention: # TODO: add cross attention implementation
    feed_forward: T5LayerFF
    
    def __init__(self, x: TensorMoments, attention: T5LayerSelfAttention, feed_forward: T5LayerFF, config: T5ConfigNNTile, cross_attention = None):
        assert not config.is_decoder or cross_attention is not None, "Cross attention must be provided for decoder blocks"
        
        self.is_decoder = config.is_decoder
        self.attention = attention
        self.feed_forward = feed_forward
        layers = attention.layers 
        if config.is_decoder:
            layers.extend(cross_attention.layers)
        layers.extend(feed_forward.layers)

        activations = attention.activations
        if config.is_decoder:
            activations.extend(cross_attention.activations[1:])
        activations.extend(feed_forward.activations[1:])
        
        super().__init__(activations, layers, config)
    
    @classmethod
    def from_torch(
        cls, 
        torch_block: T5BlockTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile, 
        next_tag: int
    ):
        assert not config.is_decoder, "TODO: add decoder block implementation"
        attention, next_tag = T5LayerSelfAttention.from_torch(torch_block.layer[0], x, config, next_tag)
        ff_layer_torch = torch_block.layer[2] if config.is_decoder else torch_block.layer[1]
        feed_forward, next_tag = T5LayerFF.from_torch(ff_layer_torch, attention.activations[-1], config, next_tag)
        block = cls(x, attention, feed_forward, config)
        return block, next_tag
    
    
class T5Stack(BaseModel):
    blocks: list[T5Block]
    
    def __init__(self, x: TensorMoments, blocks: list[T5Block], config: T5ConfigNNTile):
        self.blocks = blocks
        
        activations = blocks[0].activations + sum([block.activations[1:] for block in blocks[1:]])
        layers = sum([b.layers for b in blocks])
        super().__init__(activations, layers, config)
    
    @classmethod
    def from_torch(
        cls, 
        torch_stack: T5StackTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile,
        next_tag: int
    ):
        blocks = []
        next_inp = x
        for layer_idx in range(len(torch_stack.block)):
            torch_block = torch_stack.block[layer_idx]
            block, next_tag = T5Block.from_torch(torch_block, next_inp, config, next_tag)
            blocks.append(block)
            next_inp = block.activations[-1]
        
        stack = cls(x, blocks, config)
        return stack, next_tag
    