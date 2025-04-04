

from nntile.layer.add import Add
from nntile.layer.layer_norm import LayerNorm
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
    layer_norm: LayerNorm
    add: Add

    def __init__(self, x: TensorMoments, attention: T5Attention, layer_norm: LayerNorm, add: Add, config: T5ConfigNNTile):
        self.attention = attention
        self.layer_norm = layer_norm
        self.add = add
        
        layers = [layer_norm, attention, add]
        activations = [x] + layer_norm.activations_output + attention.activations[1:] + add.activations_output
        super().__init__(activations, layers, config)
        
    @classmethod
    def from_torch(
        cls, 
        torch_layer: T5LayerSelfAttentionTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile, 
        next_tag: int
    ):
        layer_norm, next_tag = LayerNorm.from_torch(torch_layer.layer_norm, x, next_tag)
        attention, next_tag = T5Attention.from_torch(torch_layer.attention, layer_norm.activations_output[0], layer_norm.activations_output[0], layer_norm.activations_output[0], config, next_tag)
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
        layers = [attention] + [cross_attention] if config.is_decoder else [] + [feed_forward]
        activations = [x] + attention.activations[1:] + [cross_attention.activations[1:]] if config.is_decoder else [] + feed_forward.activations[1:]
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
        feed_forward, next_tag = T5LayerFF.from_torch(ff_layer_torch, attention.activations_output[0], config, next_tag)
        block = cls(x, attention, feed_forward, config)
        return block, next_tag
    
    
class T5Stack(BaseModel):
    blocks: list[T5Block]
    
    def __init__(self, x: TensorMoments, blocks: list[T5Block], config: T5ConfigNNTile):
        self.blocks = blocks
        
        activations = [x] + sum([block.activations[1:] for block in blocks])
        layers = [b.layers for b in blocks]
        super().__init__(activations, layers, config)
    
    @classmethod
    def from_torch(
        cls, 
        torch_stack: T5StackTorch, 
        x: TensorMoments, 
        config: T5ConfigNNTile,
        next_tag: int
    ):
        blocks = [
            T5Block.from_torch(
                block, x, config, next_tag
            ) for block in torch_stack.layer
        ]
        stack = cls(x, blocks, config)
        return stack, next_tag
    
        
        
        
    