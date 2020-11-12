import tensorflow as tf
import numpy as np


def build_models(state_dim, action_dim, layer_dims=[400, 300]):
    actor = tf.keras.Sequential()
    In1 = 1/np.sqrt(layer_dims[0])
    actor.add(
        tf.keras.layers.Dense(
            layer_dims[0],
            activation='relu',
            input_shape=(state_dim,),
            # kernel_initializer=tf.random_uniform_initializer(-In1, In1),
            # bias_initializer=tf.random_uniform_initializer(-In1, In1),)
        ))
    In2 = 1/np.sqrt(layer_dims[1])
    actor.add(
        tf.keras.layers.Dense(
            layer_dims[1],
            activation='relu',
            # kernel_initializer=tf.random_uniform_initializer(-In2, In2),
            # bias_initializer=tf.random_uniform_initializer(-In2, In2))
        ))
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    actor.add(tf.keras.layers.Dense(action_dim, activation='tanh'))

    critic = tf.keras.Sequential()
    critic.add(
        tf.keras.layers.Dense(
            layer_dims[0],
            activation='relu',
            input_shape=(state_dim + action_dim,),
            kernel_initializer=tf.random_uniform_initializer(-In1, In1),
            bias_initializer=tf.random_uniform_initializer(-In1, In1)))
    critic.add(
        tf.keras.layers.Dense(
            layer_dims[1],
            activation='relu',
            kernel_initializer=tf.random_uniform_initializer(-In2, In2),
            bias_initializer=tf.random_uniform_initializer(-In2, In2)))
    critic.add(tf.keras.layers.Dense(1, activation='linear'))

    return actor, critic
