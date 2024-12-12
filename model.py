import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation, Dropout, add, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Attention block implementation
def attention_block(x, g, inter_channels):
    theta_x = Conv3D(inter_channels, (1, 1, 1), padding='same')(x)
    phi_g = Conv3D(inter_channels, (1, 1, 1), padding='same')(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(f)
    return multiply([x, psi_f])

def unet_model(input_shape, activation='relu', batch_norm=False, dropout_rate=0.0, l2_reg=0.0, attention=False):
    inputs = Input(input_shape)
    
    def conv_block(x, filters, kernel_size=(3, 3, 3), activation=activation, padding='same', l2_reg=l2_reg):
        conv = Conv3D(filters, kernel_size, padding=padding, kernel_regularizer=l2(l2_reg))(x)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        return conv

    # Encoder path
    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 256)
    if dropout_rate > 0:
        conv4 = Dropout(dropout_rate)(conv4)

    # Decoder path
    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    up5 = conv_block(up5, 128)
    if attention:
        up5 = attention_block(up5, conv3, inter_channels=64)
    merge5 = concatenate([conv3, up5], axis=-1)
    conv5 = conv_block(merge5, 128)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = conv_block(up6, 64)
    if attention:
        up6 = attention_block(up6, conv2, inter_channels=32)
    merge6 = concatenate([conv2, up6], axis=-1)
    conv6 = conv_block(merge6, 64)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = conv_block(up7, 32)
    if attention:
        up7 = attention_block(up7, conv1, inter_channels=16)
    merge7 = concatenate([conv1, up7], axis=-1)
    conv7 = conv_block(merge7, 32)

    outputs = Conv3D(3, (1, 1, 1), activation='linear')(conv7)

    model = Model(inputs=inputs, outputs=outputs)

    return model
