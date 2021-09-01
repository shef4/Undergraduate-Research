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

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(rotationAngle=90,speed=1,sepDist=10 ,dronesize=2)
agent = Agent(ALPHA=0.005, input_dims=10000, GAMMA=0.3,n_actions=4,
             layer1_size=64, layer2_size=64, fname="models/reinforce_forward_policy.h5")

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
n_episodes = 3000
#agent.load_model() 
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
    
    while not done and steps < 20:
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
            
        #TODO: 
        # add heading 
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observation_
        score += reward
        steps += 1
        steps_arr.append(steps)
        score_arr.append(score)
        action_arr.append(action_Dict[action])
        
        
        # Render will plot the state as a curve, and also plots a top down plot of the trees
        env.render(steps)
        #print('action:  %-6s ' % action_Dict[action], 'score: %6.1f ' % score,'steps: %6.1f ' % (steps-1))
    #stores the values of step for graphing
    score_history.append(score)
    steps_history.append(steps)
    agent.learn()
    ep_graph(steps_arr,score_arr,action_arr,i)
    print('ep: %6.1d' % i, ' score : %6.1f' % score,' steps : %-6.1f' % (steps-1),'average_score : %6.1f' % np.mean(score_history[-100:]))
    avg_score_history.append(np.mean(score_history[-100:]))
    if (i%500 == 0):
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

def ep_graph(steps_arr,score_arr,action_arr,ep):
    # Function to map the colors as a list from the input list of x variables
    def pltcolor(lst):
        cols=[]
        for l in lst:
            if l=='U':
                cols.append(0)
            elif l=='L':
                cols.append(1)
            elif l=='R':
                cols.append(2)
            else:
                cols.append(3)
        return cols

    colors = ["green", "blue", "cyan", "red"]
    colormap = matplotlib.colors.ListedColormap(colors)
    color_indices = pltcolor(action_arr)

    fig, ax = plt.subplots()
    ax.plot(steps_arr, score_arr,'--', color='black')
    ax.scatter(steps_arr, score_arr, c=color_indices, cmap=colormap)
    plt.title('Agent Score Vs Action Steps '+str(ep), fontsize=14)
    plt.xlabel('Num. Steps', fontsize=14)
    plt.ylabel('Agent Score', fontsize=14)
    
    gre_patch = mpatches.Patch(color='green', label='Forward')
    red_patch = mpatches.Patch(color='red', label='Back')
    pin_patch = mpatches.Patch(color='cyan', label='Turn Right')
    blu_patch = mpatches.Patch(color='blue', label='Turn Left')
    
    plt.legend(handles=[gre_patch, red_patch, blu_patch, pin_patch])
    plt.grid(True)
    plt.savefig('outputs/ep_graph/ep_0_rot'+str(ep)+'.png',transparent=False)
    plt.show()
    plt.close()
