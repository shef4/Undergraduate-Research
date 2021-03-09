# Simple script to demonstrate how to use the environment as a black box.

# Import the environment
import envV2 as ENV

#Load the environment, it has a number of variables that can be initailized.
#Here we just set the movement speed of the drone and drone size radius.
env = ENV.sonarEnv(speed=0.5,dronesize=0.1)

# This loop just moves forward for 100 steps, if the drone crashes we reset.
for i in range(100):
    
    # Step function has the action as input: 0 - forward, 1 - backwards, 2 - turn left, 3 - turn right
    # It returns a vector of the state, the reward, the `finished' boolean
    a = env.step(0)
    
    # Render will plot the state as a curve, and also plots a top down plot of the trees
    env.render(i)
    
    # If `finished' we reset
    if a[2]:
        env.reset()
    

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