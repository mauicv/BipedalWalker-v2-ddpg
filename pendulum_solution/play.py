"""Pendulum solution."""

import tensorflow as tf
import gym


class PendulumAgent:
    def __init__(self):
        self.actor, self.critic = (None, None)
        self.load_models()
        assert((self.actor, self.critic) != (None, None))

    def load_models(self):
        try:
            self.critic = tf.keras.models \
                .load_model('./pendulum_solution/critic')
            self.actor = tf.keras.models \
                .load_model('./pendulum_solution/actor')
            return True
        except Exception as err:
            print(err)

    def get_action(self, state):
        return self.actor(state)*2


def play(steps):
    env = gym.make('Pendulum-v0')
    agent = PendulumAgent()
    state = env.reset()
    agent.actor.summary()
    agent.critic.summary()
    for i in range(steps):
        action = agent.get_action(state[None])[0]
        state, reward, done, _ = env \
            .step(action)
        state = state
        env.render()


play(steps=200)
