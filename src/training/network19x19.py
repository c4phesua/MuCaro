from keras import Input, Model, Layer, ops
from keras.layers import Conv2D, Activation, Add, LayerNormalization, Reshape, MultiHeadAttention, Flatten, Dense, \
    Permute, Embedding
from keras.src.saving import register_keras_serializable

from model.network import MuCaroNetwork


@register_keras_serializable(package="Custom", name="AddPositionalEmbedding")
class AddPositionalEmbedding(Layer):
    def __init__(self, seq_length: int = 9, embed_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.pos_emb_layer = Embedding(input_dim=seq_length, output_dim=embed_dim)

    def build(self, input_shape):
        self.pos_emb_layer.build((self.seq_length,))
        super().build(input_shape)

    def call(self, inputs):
        positions = ops.arange(start=0, stop=self.seq_length, step=1)
        position_embeddings = self.pos_emb_layer(positions)
        return ops.add(inputs, position_embeddings)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_length": self.seq_length,
            "embed_dim": self.embed_dim,
        })
        return config


def residual_block(layer_x, filters, strides):
    shortcut = layer_x
    layer_x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, data_format='channels_first',
                     padding='same', use_bias=False)(layer_x)
    layer_x = LayerNormalization(axis=[1, 2, 3])(layer_x)
    layer_x = Activation(activation='silu')(layer_x)
    layer_x = Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first', padding='same')(layer_x)
    if shortcut.shape[1] != filters or strides != 1:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, data_format='channels_first',
                          padding='same')(shortcut)
    layer_x = Add()([layer_x, shortcut])
    layer_x = Activation(activation='silu')(layer_x)
    return layer_x


def get_mucaro_19x19_network():
    mu_network = MuCaroNetwork()
    h_input = Input(shape=(2, 361))
    h_x = Reshape(target_shape=(2, 19, 19))(h_input)
    h_x = residual_block(h_x, 32, strides=1)
    h_x = residual_block(h_x, 32, strides=2)
    h_x = residual_block(h_x, 64, strides=2)
    h_x = residual_block(h_x, 128, strides=2)
    h_x = Reshape(target_shape=(128, 9))(h_x)
    h_x = Permute((2, 1))(h_x)
    h_x = AddPositionalEmbedding(seq_length=9, embed_dim=128)(h_x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(
        query=h_x,
        value=h_x,
        key=h_x
    )
    h_network = Add()([h_x, attn_output])
    h_network = LayerNormalization()(h_network)
    h_network = Flatten()(h_network)
    h_network = Dense(512, activation='linear')(h_network)
    h_network = LayerNormalization()(h_network)
    mu_network.h_network = Model(inputs=h_input, outputs=h_network, name='h_network')
    mu_network.h_network.summary()
