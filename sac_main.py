import gym
import pybullet_envs
import os
import torch
import numpy as np
from sac_learn_2 import Agent
from plotting_utils import plot_learn_curve

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims= env.observation_space.shape, env = env,
                    n_actions = env.action_space.shape[0])
    games = 250
    filename = 'inverted_pendulum.png'
    fig_file = 'plots' + filename
    best_score = env.reward_range[0]
    score_hist = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode = 'human')

    for i in range(games):
        obs = env.reset()
        done = False
        score = 0 
        while not done:
            action = agent.get_actions(obs)
            observation_,reward, done, info = env.step(action)
            score += reward
            agent.save_data(obs,action,reward,observation_,done)
            if not load_checkpoint:
                agent.networks_learn()
                obs = observation_
        score_hist.append(score)
        avg_score = np.mean(score_hist[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_model()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(games)]
        plot_learn_curve(x, score_hist, fig_file)

        
