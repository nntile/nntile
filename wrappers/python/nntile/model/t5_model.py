import nntile
from nntile.model.t5_block import T5Stack, T5Block
from nntile.model.t5_config import T5ConfigNNTile, T5EncoderDecoderConfig
import copy
from nntile.tensor import TensorMoments
from transformers.models.t5.modeling_t5 import T5Model as T5ModelTorch
from nntile.model.base_model import BaseModel

class T5Model(BaseModel):
    def __init__(self, x: TensorMoments, decoder_x: TensorMoments, 
                 encoder: T5Stack, decoder: T5Stack, 
                 encoder_config: T5ConfigNNTile, decoder_config: T5ConfigNNTile
    ):
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        self.x = x
        self.decoder_x = decoder_x
        
        self.encoder = encoder
        self.decoder = decoder
        
        activations = [x] + self.encoder.activations[1:] + [decoder_x] + self.decoder.activations[1:]
        layers = self.encoder.layers + self.decoder.layers
        
        super().__init__(activations, layers, T5EncoderDecoderConfig(encoder_config, decoder_config))

    @classmethod
    def from_torch(cls, torch_model: T5ModelTorch, x: TensorMoments, decoder_x: TensorMoments, config: T5ConfigNNTile, next_tag: int=0):
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        
        encoder, next_tag = T5Stack.from_torch(torch_model.encoder, x, encoder_config, next_tag=next_tag)
        encoder_output = encoder.activations[-1]
        decoder, next_tag = T5Stack.from_torch(torch_model.decoder, decoder_x, decoder_config, next_tag=next_tag, encoder_output=encoder_output)

        return cls(x, decoder_x, encoder, decoder, encoder_config, decoder_config), next_tag
