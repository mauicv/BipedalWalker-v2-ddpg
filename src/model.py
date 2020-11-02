import tensorflow as tf


def build_models(state_dim, action_dim):
    actor = tf.keras.Sequential()
    actor.add(tf.keras.layers.Dense(256,
                                    activation='relu',
                                    input_shape=(state_dim,)))
    actor.add(tf.keras.layers.Dense(256, activation='relu'))
    actor.add(tf.keras.layers.Dense(action_dim, activation='tanh'))

    critic = tf.keras.Sequential()
    critic.add(tf.keras.layers.Dense(256,
                                     activation='relu',
                                     input_shape=(state_dim + action_dim,)))
    critic.add(tf.keras.layers.Dense(256, activation='relu'))
    critic.add(tf.keras.layers.Dense(1, activation='linear'))

    return actor, critic
