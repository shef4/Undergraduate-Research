# Simple script to demonstrate how to use the environment as a black box.

# Import the environment
import envV2 as ENV
#Import agent + policy model : Reinforce
from Reinforce_keras import Agent
# other dependencies
import matplotlib.pyplot as plt
import numpy as np

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(rotationAngle=90,speed=1,sepDist=3,dronesize=0.1)

agent = Agent(ALPHA=0.0005, input_dims=10000, GAMMA=0.99,n_actions=4,
             layer1_size=64, layer2_size=64)

score_history = []
steps_history = []
stepDirct_history = [0,0,0,0]

#print observation
n_episodes = 1000
   
for i in range(n_episodes):
    done=False
    score = 0
    steps=0
    observation = env.reset()
    strAction = ''
    #env.setPrefAction()
    #prefActionEp.append(env.getPrefAction)
    
    while not done and steps < 10 :
        #input: get G for profomance - array len episode save G zero
        action = agent.choose_action(observation)
        
        #next steps: 
        # - Reward shaping and obstical avoidance
        # - model shaping based off input shape (Fourier series of tree echo's) 
        #   might filter input audio to make differences stand out 
        #   Qu: what differece are there in audio coming from 
        #       the front vs the back vs the sides?
        # - Qu: What range of directions does the echo inout come from?
        stepDirct_history[action] += 1
        if action == 0:
            strAction = 'U'
        elif action == 1:
            strAction = 'D'
        elif action == 2:
            strAction = 'L'
        elif action == 3:
            strAction = 'R'
            
        #TODO: 
        # add heading 
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observation_
        score += reward
        steps += 1
        
        # Render will plot the state as a curve, and also plots a top down plot of the trees
        #env.render(steps)
        print('action: ', strAction, 'score %.1f: ' % score,'steps %.1f: ' % (steps-1))
    #stores the values of step for graphing
    score_history.append(score)
    steps_history.append(steps)
    agent.learn()
    
    print('episode: ', i, 'score %.1f:' % score,'steps %.1f:' % (steps-1),'average_score %.1f:' % np.mean(score_history[-100:]))
    
agent.save_model()

plt.plot(score_history[:25:])
#plt.plot(steps_history[:25:])
plt.plot(stepDirct_history[:])
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#act = [0,1,2,3]
#ax.bar(act, stepDirct_history[0:4])
plt.show()
# Frees some memory when finished with the environment
env.close()
"""
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

"""