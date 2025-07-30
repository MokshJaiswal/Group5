import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense, Embedding
from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.regularizers import l2
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import KFold
#from time import time

# 1. Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        import numpy as np  # Move back to top if you uncomment permanently
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


# 2. Transformer Block with Causal Masking
class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        causal_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        attn_output = self.mha(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    
def build_causal_transformer(vocab_size, max_len, d_model=32, num_heads=2, ff_dim=32, rate=0.1):
    inputs = tf.keras.Input(shape=(max_len,))
    
    # Learnable Embeddings
    x = layers.Embedding(vocab_size, d_model, mask_zero=True)(inputs)
    
    # Positional Encoding (assuming you have this class defined)
    x = PositionalEncoding(max_len, d_model)(x)
    x = layers.Dropout(rate)(x)
    
    # Transformer Blocks
    x = TransformerBlock(d_model, num_heads, ff_dim, rate)(x, training=True)
    x = TransformerBlock(d_model, num_heads,ff_dim, rate)(x, training=True)
    
    # Pooling and Output
    x = layers.GlobalAveragePooling1D()(x)  # Fixed typo: "GlObal" → "Global"
    x = layers.Dropout(rate)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(rate)(x)
    outputs = layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    return Model(inputs=inputs, outputs=outputs)