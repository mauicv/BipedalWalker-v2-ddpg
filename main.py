import gym
from src.agent import Agent
import click
import os
import shutil
from time import time
import numpy as np

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
    logger = Logging([
        'episode',
        'rewards',
        'episode_length',
        'epsiode_run_time',
        'average_step_run_time'
    ])

    env, state_space_dim, action_space_dim = setup_env()
    replay_buffer = ReplayBuffer(state_space_dim=state_space_dim,
                                 action_space_dim=action_space_dim,
                                 size=1000000)

    agent = Agent(state_space_dim,
                  action_space_dim,
                  low_action=-1,
                  high_action=1,
                  exploration_value=0.2,
                  tau=0.05,
                  load=True)

    train = Train(discount_factor=0.99,
                  actor_learning_rate=0.00005,
                  critic_learning_rate=0.0005)

    for episode in range(episodes):
        state = np.array(env.reset(), dtype='float32')
        episode_reward = 0
        step_count = 0
        done = False
        episode_start_time = time()
        step_times = []
        while not done and step_count < steps:
            step_time_start = time()
            step_count += 1

            # training code
            action = agent.get_action(state[None], with_exploration=True)[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, next_state, action, reward, done)
            if replay_buffer.ready:
                states, next_states, actions, \
                    rewards, dones = replay_buffer.sample()
                train(agent, states, next_states, actions, rewards, dones)
                agent.track_weights()
            state = next_state

            episode_reward += reward
            step_time_end = time()
            step_times.append(step_time_end - step_time_start)
        episode_end_time = time()
        epsiode_time = episode_end_time - episode_start_time
        average_step_time = np.array(step_times).mean()
        logger.log([
            episode,
            episode_reward,
            step_count,
            epsiode_time,
            average_step_time,
        ])
        agent.save_models()


@cli.command()
@click.pass_context
@click.option('--steps', '-s', default=100, type=int,
              help='Max number of steps per episode')
@click.option('--noise', '-n', is_flag=True,
              help='With exploration')
def play(ctx, steps, noise):
    env, state_space_dim, action_space_dim = setup_env()
    agent = Agent(state_space_dim,
                  action_space_dim,
                  low_action=-1,
                  high_action=1,
                  exploration_value=0.2,
                  load=True)
    state = env.reset()
    for i in range(steps):
        action = agent.get_action(state[None], with_exploration=noise)[0]
        state, reward, done, _ = env \
            .step(action)
        env.render()


@cli.command()
@click.pass_context
def clean(ctx):
    for save_path in os.listdir('save'):
        path = f'./save/{save_path}'
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


if __name__ == '__main__':
    cli()
