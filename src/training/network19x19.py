import torch
from keras import Input, Model, Layer, ops
from keras.layers import Conv2D, Activation, Add, LayerNormalization, Reshape, MultiHeadAttention, Flatten, Dense, \
    Permute, Embedding, GroupNormalization, Concatenate
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
    layer_x = GroupNormalization(axis=1, groups=1)(layer_x)
    layer_x = Activation(activation='silu')(layer_x)
    layer_x = Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first', padding='same', use_bias=False)(
        layer_x)
    layer_x = GroupNormalization(axis=1, groups=1)(layer_x)
    if shortcut.shape[1] != filters or strides != 1:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, data_format='channels_first',
                          padding='same')(shortcut)
    layer_x = Add()([layer_x, shortcut])
    layer_x = Activation(activation='silu')(layer_x)
    return layer_x


def get_mucaro_19x19_network():
    mu_network = MuCaroNetwork()
    # H network setup
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
    h_layers = Add()([h_x, attn_output])
    h_layers = LayerNormalization()(h_layers)
    h_layers = Flatten()(h_layers)
    h_layers = Dense(512, activation='linear')(h_layers)
    h_layers = LayerNormalization()(h_layers)
    mu_network.h_network = Model(inputs=h_input, outputs=h_layers, name='h_network')
    mu_network.h_network.summary()

    # F network setup
    f_input = Input(shape=(512,))
    v_head = Dense(256, use_bias=False)(f_input)
    v_head = LayerNormalization()(v_head)
    v_head = Activation(activation='silu')(v_head)
    v_head = Dense(32, use_bias=False)(v_head)
    v_head = LayerNormalization()(v_head)
    v_head = Activation(activation='silu')(v_head)
    v_head = Dense(1, kernel_initializer='zeros', activation='tanh')(v_head)

    p_head = Dense(512, use_bias=False)(f_input)
    p_head = LayerNormalization()(p_head)
    p_head = Activation(activation='silu')(p_head)
    p_head = Dense(361, kernel_initializer='zeros', activation='linear')(p_head)
    mu_network.f_network = Model(inputs=f_input, outputs=[p_head, v_head], name='f_network')
    mu_network.f_network.summary()

    # G network setup
    g_latent_input = Input(shape=(512,))
    g_action_input = Input(shape=(1,), dtype=torch.long)
    g_action_embedding = Embedding(input_dim=361, output_dim=64)(g_action_input)
    g_action_embedding = Flatten()(g_action_embedding)
    g_layers = Concatenate()([g_latent_input, g_action_embedding])
    r_head = Dense(256, use_bias=False)(g_layers)
    r_head = LayerNormalization()(r_head)
    r_head = Activation(activation='silu')(r_head)
    r_head = Dense(32, use_bias=False)(r_head)
    r_head = LayerNormalization()(r_head)
    r_head = Activation(activation='silu')(r_head)
    r_head = Dense(1, kernel_initializer='zeros', activation='tanh')(r_head)

    delta_latent = Dense(512, use_bias=False)(g_layers)
    delta_latent = LayerNormalization()(delta_latent)
    delta_latent = Activation(activation='silu')(delta_latent)
    delta_latent = Dense(512, use_bias=False)(delta_latent)
    next_latent = Add()([delta_latent, g_latent_input])
    next_latent = LayerNormalization()(next_latent)
    mu_network.g_network = Model(inputs=[g_latent_input, g_action_input], outputs=[next_latent, r_head],
                                 name='g_network')
    mu_network.g_network.summary()
    mu_network.build_consistency_networks(512)
    mu_network.verify_contract((2, 361), 361)
    return mu_network