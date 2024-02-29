from src.transformer.positionalencoding import PositionalEncoding
from src.transformer.positionalwisefeedforward import Positionalwisefeedforward

# position_enoc=Positional_encoding()

import tensorflow as tf

# Assuming the classes PositionWiseFeedForward and PositionalEncoding are defined as provided

# Create a dummy input tensor with shape (batch_size, seq_length, d_model)
# For this example, let's assume a batch size of 64, sequence length of 50, and d_model of 512
dummy_input = tf.random.uniform((64, 50, 512), dtype=tf.float32)

# Instantiate the PositionWiseFeedForward layer
d_model = 512
d_ff = 2048
pwff_layer = Positionalwisefeedforward(d_model, d_ff)

# Apply the PositionWiseFeedForward layer to the dummy input
pwff_output = pwff_layer(dummy_input)
print(f"Output shape after PositionWiseFeedForward: {pwff_output.shape}")

# Instantiate the PositionalEncoding layer
position = 50  # This should match or exceed the sequence length in your input
pos_enc_layer = PositionalEncoding(position, d_model)

# Apply the PositionalEncoding layer to the dummy input
pos_enc_output = pos_enc_layer(dummy_input)
print(f"Output shape after PositionalEncoding: {pos_enc_output.shape}")

