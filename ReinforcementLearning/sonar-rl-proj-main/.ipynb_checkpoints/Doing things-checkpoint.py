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
import time

filename = "reinforce_7L_Weights_final"

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
agent = Agent(ALPHA=0.002, GAMMA=0.70, input_dims=10000, n_actions=4,load = True, fname_policy=str("models/"+filename+"_policy.h5"))

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
n_steps = 30
env = ENV.sonarEnv(rotationAngle=90)


for i in range(n_episodes):
    done=False
    score = 0
    steps = 0
    observation = env.reset()
    strAction = ''
    #env.setPrefAction()
    #prefActionEp.append(env.getPrefAction)
    steps_arr = [0]
    score_arr = [0]
    action_arr = ['U']
    #start = time.time() #4sec-5steps 11sec-20steps 21sec-30steps
    while not done and steps < n_steps:
        #time 
        action = agent.choose_action(observation)
        
        stepDirct_history[action] += 1
            
         #time 
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation,action, reward)
        observation = observation_
        
        score += reward
        steps += 1
        steps_arr.append(steps)
        score_arr.append(score)
        action_arr.append(action_Dict[action])
        if(i%500 == 0):
            env.render(steps,i,reward)
        
        #print('action:  %-6s ' % action_Dict[action], 'score: %6.1f ' % score,'steps: %6.1f ' % (steps-1))
    #end = time.time()
    #print(end - start,"\n")
    score_history.append(score)
    steps_history.append(steps)
    agent.learn()
    if(i%10== 0):
        plt.bar([1,2,3,4], height = stepDirct_history)
        plt.xticks([1,2,3,4], ['Forwards','backwards','T Left', 'T Right'])
        plt.savefig('outputs/ep_graph/Action_Prob_Dist_'+str(i)+'.png',transparent=False)
        agent.ep_graph(steps_arr,score_arr,action_arr,i, np.mean(score_history[-10:]))
    print('ep: %6.1d' % i, ' score : %6.1f' % score,' steps : %-6.1f' % (steps-1),
          'average_score : %5.2f' % np.mean(score_history[-10:]),
          'F:%4.1f' % (stepDirct_history[0]/sum(stepDirct_history)) ,
          'B:%4.1f' % (stepDirct_history[1]/sum(stepDirct_history)) ,
          'L:%4.1f' % (stepDirct_history[2]/sum(stepDirct_history)) ,
          'R:%4.1f' %(stepDirct_history[3]/sum(stepDirct_history)) )
    
    stepDirct_history = [0,0,0,0] 
    avg_score_history.append(np.mean(score_history[-10:]))
    if (i%500 == 0):
        agent.save_model()
        



env.plot_results(avg_score_history, steps_arr,filename)
# Frees some memory when finished with the environment
env.close()