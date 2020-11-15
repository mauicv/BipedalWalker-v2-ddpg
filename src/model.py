import tensorflow as tf


def build_models(state_dim, action_dim, layer_dims=[400, 300], upper_bound=2):
    actor = get_actor(state_dim, action_dim, layer_dims, upper_bound)
    critic = get_critic(state_dim, action_dim, layer_dims)
    return actor, critic


def get_actor(state_dim, action_dim, layer_dims, upper_bound):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    out = tf.keras.layers.Dense(
        layer_dims[0], activation="relu",
        )(inputs)
    out = tf.keras.layers.Dense(
        layer_dims[1], activation="relu",
        )(out)
    outputs = tf.keras.layers.Dense(
        action_dim, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_dim, action_dim, layer_dims):
    state_input = tf.keras.layers.Input(shape=(state_dim))
    action_input = tf.keras.layers.Input(shape=(action_dim))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    out = tf.keras.layers.Dense(layer_dims[0], activation="relu")(concat)
    out = tf.keras.layers.Dense(layer_dims[1], activation="relu")(out)
    outputs = tf.keras.layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)
