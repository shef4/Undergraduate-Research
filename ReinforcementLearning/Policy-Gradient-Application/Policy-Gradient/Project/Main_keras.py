# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:55:11 2020

@author: sefun
"""
from Reinforce_keras import Agent
from Continous_grid_world import CGridWorld

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    
    env = CGridWorld(size = 3, p = 0.5)
    
    agent = Agent(ALPHA=0.0005, input_dims=3, GAMMA=0.99,n_actions=4,
                  layer1_size=64, layer2_size=64)

    score_history = []
    prefActionEp = []
    
    #print observation
    n_episodes =500
   
    for i in range(n_episodes):
        done=False
        score = 0
        steps=0
        observation = env.reset()
        env.setPrefAction()
        prefActionEp.append(env.getPrefAction)
        
        while not done and steps < 1000:
            #not done
            action = agent.choose_action(observation)
            
            
            if action == 1:
                strAction = 'U'
            elif action == 2:
                strAction = 'D'
            elif action == 3:
                strAction = 'L'
            elif action == 4:
                strAction = 'R'
            
            observation_, reward, done, info = env.step(strAction)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
            steps += 1
            
        
        score_history.append(score)
        
        agent.learn()
        
        print('episode: ', i, 'prefered Action:', env.getPrefAction(),'score %.1f:' % score,'steps %.1f:' % (steps-1),
              'average_score %.1f:' % np.mean(score_history[-100:]))
    agent.save_model()
    
    plt.plot(score_history[:10:])
    
    
    plt.plot(steps[:10:])
    plt.show()