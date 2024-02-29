import math
import sys
import tensorflow as tf 
from tensorflow.keras import layers,models
from tensorflow.keras.activations import relu,tanh
from tensorflow.keras.optimizers import Adam,Adagrad
from src.logger import logging
from src.exception import CustomException




class Positionalwisefeedforward(tf.keras.layers.Layer):
     def __init__(self,d_model,d_ff):
          logging.info("Calling the constructor of  base class which is layers.Layer")
          super(Positionalwisefeedforward,self).__init__()
          logging.info('Calling the Constructor Successfully ')
          logging.info('<===================================>')
          self.fc1=tf.keras.layers.Dense(d_ff,activation='relu')
          logging.info("Make the hidden layer variable of feed forward ")
          self.fc2=tf.keras.layers.Dense(d_model)
          logging.info("Make the input/output feed forward Dense layer ")
          
          
          
          
     def call(self,x):
          try:
               logging.info("Calling the call method successfully ")
               x=self.fc1(x)
               logging.info("input is passed through the first hidden Dense layer ")
               logging.info("Returning the output from second Dense layer of output model dimension ")
               return self.fc2(x)
          except Exception as e:
               raise CustomException(e,sys) from e
