
# coding: utf-8

# In[13]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

sns.set_context('poster')
get_ipython().magic(u'matplotlib inline')

grid = np.array([[-1,-1,-1,-1], [0, -1, -1, -1], [-1,-1,-1,-1], [-1, -1, -1, -1]])
sns.heatmap(grid, cmap='RdBu',linewidths=.05, annot=True, cbar=False, annot_kws={"size":14})
plt.title('Rewards')

#the states of the grid
states = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#the actions the agent can take
actions = ['up', 'down', 'left', 'right']
#the discounting factor
gamma = 1

nactions = len(actions)
nstates = len(states) 

#the policy for a random walk
policy = np.ones((nstates, nactions)) / nactions

#the transition function from state s with action a (16, 4)-size
transitions = np.array([[ 0,  4,  0,  1],
       [ 1,  5,  0,  2],
       [ 2,  6,  1,  3],
       [ 3,  7,  2,  3],
       [ 4,  4,  4,  4],
       [ 1,  9,  4,  6],
       [ 2,  10,  5,  7],
       [ 3,  11,  6,  7],
       [ 4,  12,  8,  9],
       [ 5,  13,  8,  10],
       [ 6,  14,  9,  11],
       [ 7,  15,  10,  11],
       [ 8,  12,  12,  13],
       [ 9,  13,  12,  14],
       [ 10,  14,  13,  15],
       [ 11,  15,  14,  15]])

#policy evaluation iteration by dynamic programming
def iterate(policy, nstates, transitions, V, gamma, delta):
    #copy old state-values into new table 
    V_prime = np.copy(V)
    #cycle through each state
    for s in range(nstates):
        v = V_prime[s]
        V_prime[s] = 0
        #cycle through each action from current state and evaluate state-value
        for a, a_prob in enumerate(policy[s]):
            reward = np.ravel(grid)[s]
            s_prime = transitions[s, a]
            
            V_prime[s] += a_prob * (reward + gamma * V[s_prime])
        #diffrence between old and new state value
        #if diffrence is small enough we have our approximation
        delta = max(delta, abs(v - V_prime[s]))
    return delta, V_prime

#policy evaluation
def evaluate_policy(policy, nstates, transitions, theta=0.0000000001, gamma=1):
    V = np.zeros(nstates)
    k = 0
    while True:
        delta = 0
        delta, V = iterate(policy, nstates, transitions, V, gamma, delta)
        print delta
        if delta < theta:
            break
        k += 1
        if k >10000:
            break
    return V

def policy_improvement(transitions, nstates, nactions, policy_eval_fn=evaluate_policy, gamma=1.0):
    
    policy = np.ones((nstates, nactions)) / nactions
    
    while True:
        #Evaluate current policy
        V = evaluate_policy(policy, nstates, transitions)
        policy_stable = True
        
        for s in range(nstates):
            b = np.argmax(policy[s])
            
            action_values = np.zeros(nactions)
            reward = np.ravel(grid)[s]
            
            for a in range(nactions):
                s_prime = transitions[s, a]
                action_values[a] += reward + gamma * V[s_prime]
            best_a = np.argmax(action_values)
            
            if b != best_a:
                policy_stable = False
            policy[s] = np.eye(nactions)[best_a]
            
        if policy_stable:
            return policy, V    


# In[14]:


policy, V = policy_improvement(transitions, nstates, nactions)


# In[15]:


V.reshape(4,4)


# In[16]:


print np.reshape(np.argmax(policy, axis=1), np.shape(grid))
print "up=0, down=1, left=2, right=3"

