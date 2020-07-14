# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:55:11 2020

@author: sefun
"""

import matplotlib.pyplot as plt
import numpy as np
from Reinforce_keras import Agent
from Continous_grid_world import CGridWorld


if __name__ == '__main__':
    
    prefAction = [1, 2, 3, 4]
    
    direction = np.random.choice(prefAction)
    
    env = CGridWorld(size = 3, p = 0.5, prefAction = direction)
    
    agent = Agent(ALPHA=0.0005, input_dims=3, GAMMA=0.99,n_actions=4,
                  layer1_size=64, layer2_size=64)

    score_history = []
    
    #print observation
    n_episodes = 200000
   
    for i in range(n_episodes):
        done=False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            
            
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        
        score_history.append(score)
        
        agent.learn()
        
        print('episode ', i, 'score %.1f' % score,
              'average_score %.1f' % np.mean(score_history[-100:]))
    
    
    plt.plot(score_history)
    plt.show()