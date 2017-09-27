import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
    sys.path.append("../")
    
from collections import defaultdict
from windyGrid import WindyGridworldEnv
from BlackjackEnv import BlackjackEnv
from cliffwalking import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()


def sarsa(env, nepisodes, gamma=1.0, alpha=0.8):
    epsilon = 0.1
    global_time_step = 1
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nepisodes),
        episode_rewards=np.zeros(nepisodes))
    
    def policy_fn(observation):
        A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    
    policy = policy_fn
    
    for i_episode in range(nepisodes):
        
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            epsilon *= 0.992
            s_prime, reward, done, _ = env.step(action)
            
            a_prime_probs = policy(s_prime)
            a_prime = np.random.choice(np.arange(len(a_prime_probs)), p=a_prime_probs)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            Q[state][action] += alpha * (reward + gamma*Q[s_prime][a_prime] - Q[state][action])
            
            global_time_step += 1
            
            if done:
                break
                
            action = a_prime
            state = s_prime
            
    return Q, stats          


Q, stats = sarsa(env, 500)

plotting.plot_episode_stats(stats)