# Simple script to demonstrate how to use the environment as a black box.

# Import the environment
import envV2 as ENV
#Import agent + policy model : Reinforce
from Reinforce_keras import Agent
# other dependencies
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

filename = "reinforce_7l-Sig-Relu-Sig-wC_f-lr-b_model"

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(rotationAngle=90,speed=1,sepDist=10 ,dronesize=2)
agent = Agent(ALPHA=0.001, GAMMA=0.9, input_dims=10000, n_actions=4,load = False, fname_policy=str("models/"+filename+"_policy.h5"),fname_predict=str("models/"+filename+"_predict.h5"))

score_history = []
avg_score_history = []
steps_history = []
stepDirct_history = [0,0,0,0]
action_Dict = {
    0 : 'U',
    1 : 'D',
    2 : 'L',
    3 : 'R'
}



#print observation
n_episodes = 10000
n_steps = 20

for i in range(n_episodes):
    done=False
    score = 0
    steps = 0
    observation,compass = env.reset()
    strAction = ''
    #env.setPrefAction()
    #prefActionEp.append(env.getPrefAction)
    steps_arr = [0]
    score_arr = [0]
    action_arr = ['U']
    
    while not done and steps < n_steps:
        #time 
        action = agent.choose_action(observation,compass)
        
        stepDirct_history[action] += 1
            
         #time 
        observation_,compass_, reward, done, info = env.step(action)
        agent.store_transition(observation,compass, action, reward)
        observation = observation_
        compass = compass_
        
        score += reward
        steps += 1
        steps_arr.append(steps)
        score_arr.append(score)
        action_arr.append(action_Dict[action])
        if(i%500 == 0):
            env.render(steps,i,reward)
        #print('action:  %-6s ' % action_Dict[action], 'score: %6.1f ' % score,'steps: %6.1f ' % (steps-1))
    score_history.append(score)
    steps_history.append(steps)
    agent.learn()
    agent.ep_graph(steps_arr,score_arr,action_arr,i)
    print('ep: %6.1d' % i, ' score : %6.1f' % score,' steps : %-6.1f' % (steps-1),'average_score : %6.1f' % np.mean(score_history[-100:]))
    avg_score_history.append(np.mean(score_history[-10:]))
    if (i%500 == 0):
        agent.save_model()
        



env.plot_results(avg_score_history,filename)
# Frees some memory when finished with the environment
env.close()