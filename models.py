"""
models.py
---------
Blueprints for the Tabular Generative Adversarial Network (GAN).
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim: int, features_shape: int) -> Sequential:
    """Generates synthetic tabular feature profiles."""
    return Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(features_shape, activation='tanh')
    ], name="Generator")

def build_discriminator(features_shape: int) -> Sequential:
    """Evaluates the authenticity of feature metrics."""
    model = Sequential([
        Dense(512, input_dim=features_shape),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ], name="Discriminator")
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def assemble_gan(generator: Sequential, discriminator: Sequential, latent_dim: int) -> Model:
    """Combines models into a locked adversarial loop."""
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output, name="GAN_Adversarial_Engine")
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan