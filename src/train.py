"""Implements Train Class

The methods are agnostic about the environment or models.
"""

import tensorflow as tf


class Train:
    def __init__(
            self,
            discount_factor=0.99,
            actor_learning_rate=0.00001,
            critic_learning_rate=0.00001):
        self.discount_factor = discount_factor
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.actor_opt = tf.keras.optimizers\
            .Adam(learning_rate=self.actor_learning_rate)
        self.critic_opt = tf.keras.optimizers\
            .Adam(learning_rate=self.critic_learning_rate)

    @tf.function
    def __call__(self, agent, states, next_states, actions, rewards, dones):
        """Runs ddpg algorithm on agent actor and critic networks."""
        actor_variables = agent.actor.trainable_variables
        critic_variables = agent.critic.trainable_variables

        # update critic
        with tf.GradientTape() as critic_tape:
            next_actions = agent.target_actor(next_states)
            Q_input = tf.concat([next_states, next_actions], axis=1)
            y = rewards[:, None] + self.discount_factor*(1 - dones) * \
                agent.target_critic(Q_input)
            Q_input = tf.concat([states, actions], axis=1)
            td_error = tf.stop_gradient(y) - agent.critic(Q_input)
            squared_error = tf.pow(td_error, 2)
            Q_loss = tf.reduce_mean(squared_error)
        critic_grads = critic_tape.gradient(Q_loss, critic_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_variables))

        # update actor
        with tf.GradientTape() as actor_tape:
            actions = agent.actor(states)
            Q_input = tf.concat([states, actions], axis=1)
            policy_loss = -tf.reduce_mean(agent.critic(Q_input))
        actor_grads = actor_tape.gradient(policy_loss, actor_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_variables))

        # return Q_loss.numpy(), policy_loss.numpy()
