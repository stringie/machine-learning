import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
    sys.path.append("../")
    
from collections import defaultdict
from cliffwalking import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()


def q_lambda(env, nepisodes, gamma=1.0, alpha=0.5, lmbd = 0.5):
    epsilon = 0.1
    global_time_step = 1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    E = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nepisodes),
        episode_rewards=np.zeros(nepisodes))    
    
    def policy_fn(observation):
        A = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    
    # The policy we're following
    policy = policy_fn
    
    for i_episode in range(nepisodes): 
        # Reset the environment
        state = env.reset()
        
        for t in itertools.count():
            epsilon *= 0.99
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            best_next_action = np.argmax(Q[next_state])
            if next_action == best_next_action:
                best_next_action = next_action
                
            delta = reward + gamma * Q[next_state][best_next_action] - Q[state][action]
            E[state][action] += 1
            
            for state in range(env.nS):
                for action in range(env.nA):
                    Q[state][action] += alpha * delta * E[state][action]
                    if next_action == best_next_action:
                        E[state][action] *= gamma * lmbd
                    else:
                        E[state][action] = 0
            
            if done:
                break
            
            global_time_step += 1    
            action = next_action
            state = next_state
    
    return Q, stats

Q, stats = q_lambda(env, 200)
print Q[36][0]

plotting.plot_episode_stats(stats)