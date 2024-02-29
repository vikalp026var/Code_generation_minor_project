import math
import tensorflow as tf
from tensorflow.keras import layers
from src.logger import logging

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        logging.info("Initializing the PositionalEncoding layer.")
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        logging.info("Calculated angles for PositionalEncoding.")
        return position * angles

    def positional_encoding(self, position, d_model):
        logging.info("Creating positional encoding matrix.")
        position = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles_rads = self.get_angles(position, i, d_model)
        
        sines = tf.math.sin(angles_rads[:, 0::2])
        cosines = tf.math.cos(angles_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        logging.info("Positional encoding matrix created.")
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        logging.info("Applying PositionalEncoding to input.")
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
