# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:53:24 2020

@author: sefun
"""

import gym
import numpy as np 

#1. Load Enviroment and Q-table

env = gym.make('FrozenLake8x8-v0')
#Q is a 2D arry numObservations=numRow numAction=numColomn
Q = np.zeros([env.observation_space.n,env.action_space.n])


#2. Parameters of Q-learning

eta = .628
g = .9
episodes = 10000000
rev_list = [] # rewards per episode calculate


#3. Q-learning Algorithm
for i in range(episodes):
    
    # Reset enviroment
    s= env.reset()
    rAll = 0
    d = False
    j = 0
    
    # Q-Table Learning algorithm
    while j < 9999:
        #env.render()
        j += 1
        # select action from Q
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # get new state & reward from enviroment
        s1,r,d,_ = env.step(a)
        # update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + g*np.max(Q[s1,:])-Q[s,a])
        
        rAll += r
        s = s1
        if d==True:
            break
    rev_list.append(rAll)
    #env.render()

rewardSum = "Reward Sum on all episodes " + str(sum(rev_list)/episodes)
print(rewardSum)
print("Final Values Q")
print(Q)

#see model play with updated Q table
# Reset environment
s = env.reset()
d = False
# The Q-Table learning algorithm
while d != True:
    env.render()
    # Choose action from Q table
    a = np.argmax(Q[s,:]) #+ np.random.randn(1,env.action_space.n)*(1./(i+1)))
    #Get new state & reward from environment
    s1,r,d,_ = env.step(a)
    #Update Q-Table with new knowledge
    #Q[s,a] = Q[s,a] + eta*(r + g*np.max(Q[s1,:]) - Q[s,a])
    s = s1
# Code will stop at d == True, and render one state before it'''