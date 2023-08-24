import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from layer import GroupNormalization

def spatial_attention(x):
    se_output = x
    maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(se_output)
    avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(se_output)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    SA = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    x = multiply([se_output, SA])
    return x
# Transformer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def Attention_Net(input_tensor):
    x = input_tensor
    x1 = squeeze_excite_block(x)
    x2 = spatial_attention(x)
    x1 = Multiply()([x, x1])
    x = Multiply()([x2, x1])
    return x

def gACNN(input_tensor):
    x = input_tensor
    channel_axis = 1
    filter = x.shape[channel_axis]
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filter//4, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    # x = Multiply()([x, input_tensor])
    return x

def local_attention(input_tensor):
    a1, a2 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(input_tensor)
    a3, a4 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(a1)
    a5, a6 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(a2)
    x = Conv2D(64, (3, 3), padding='same')(a3)
    x = BatchNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x1 = x = Activation(gelu)(x)
    x1 = Attention_Net(x1)
    x1 = Multiply()([x1, x])
    a3 = concatenate([x, x1])

    x = Conv2D(64, (3, 3), padding='same')(a4)
    x = BatchNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x1 = x = Activation(gelu)(x)
    x1 = Attention_Net(x1)
    x1 = Multiply()([x1, x])
    a4 = concatenate([x, x1])

    x = Conv2D(64, (3, 3), padding='same')(a5)
    x = BatchNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x1 = x = Activation(gelu)(x)
    x1 = Attention_Net(x1)
    x1 = Multiply()([x1, x])
    a5 = concatenate([x, x1])

    x = Conv2D(64, (3, 3), padding='same')(a6)
    x = BatchNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x1 = x = Activation(gelu)(x)
    x1 = Attention_Net(x1)
    x1 = Multiply()([x1, x])
    a6 = concatenate([x, x1])
    a = concatenate([a3, a4, a5, a6])

    return a
def CBAM(input_tensor):
    x = input_tensor
    x1 = squeeze_excite_block(x)
    x = Multiply()([x1, x])
    x1 = spatial_attention(kernel_size=3)(x)
    x = Multiply()([x1, x])

    return x

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf





