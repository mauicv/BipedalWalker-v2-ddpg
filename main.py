import gym
from src.agent import Agent
import click
import os
import shutil

from src.memory import ReplayBuffer
from src.train import Train
from src.logging import Logging

ENV_NAME = 'BipedalWalker-v3'


def setup_env():
    env = gym.make(ENV_NAME)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    return env, state_space_dim, action_space_dim


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
@click.pass_context
@click.option('--episodes', '-e', default=500, type=int,
              help='Number of epsiodes of training')
@click.option('--steps', '-s', default=200, type=int,
              help='Max number of steps per episode')
def train(ctx, episodes, steps):
    logger = Logging(['episode', 'rewards', 'episode_length'])

    env, state_space_dim, action_space_dim = setup_env()
    replay_buffer = ReplayBuffer(state_space_dim=state_space_dim,
                                 action_space_dim=action_space_dim,
                                 size=10000)

    agent = Agent(state_space_dim,
                  action_space_dim,
                  low_action=-1,
                  high_action=1,
                  load=True)

    train = Train(discount_factor=0.99,
                  actor_learning_rate=0.00001,
                  critic_learning_rate=0.00001)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        while not done and step_count < steps:
            step_count += 1
            action = agent.get_action(state[None], with_exploration=True)[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, next_state, action, reward, done)
            episode_reward += reward
            if replay_buffer.ready:
                states, next_states, actions, \
                    rewards, dones = replay_buffer.sample()
                train(agent, states, next_states, actions, rewards, dones)
                agent.track_weights()

        logger.log([episode, episode_reward, step_count])
        agent.save_models()


@cli.command()
@click.pass_context
@click.option('--steps', '-s', default=1000, type=int,
              help='Max number of steps per episode')
def play(ctx, steps):
    env, state_space_dim, action_space_dim = setup_env()
    agent = Agent(state_space_dim,
                  action_space_dim,
                  low_action=-1,
                  high_action=1,
                  load=True)
    state = env.reset()
    for i in range(steps):
        action = agent.get_action(state[None])[0]
        state, reward, done, _ = env \
            .step(action)
        env.render()


@cli.command()
@click.pass_context
def clean(ctx):
    for loc in os.listdir('save'):
        shutil.rmtree(f'/save/{loc}')


if __name__ == '__main__':
    cli()
