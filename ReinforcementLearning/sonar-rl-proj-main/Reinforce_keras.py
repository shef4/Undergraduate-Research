# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:31:48 2020

@author: sefun
"""
from keras.layers import Dense,Activation, Input, concatenate, Flatten, Dropout
from keras.models import Model, load_model
from tensorflow.keras import activations
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99,n_actions=4, input_dims=128,load = False,
                 fname_policy=None,fname_predict=None):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [0, 1, 2, 3]
        
        self.state_memory = []
        self.compass_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        self.load = load
        self.model_file_policy = str(fname_policy)
        self.model_file_predict = str(fname_predict)
        self.policy, self.predict = self.build_policy_network()
        self.policy, self.predict = self.load_m()
        
        
    def build_policy_network(self):
        #env2d = Input(shape=(self.input_dims,self.input_dims))

        '''
        4 layer dense relu
        dense1 = Dense(self.fc1_dims, activation='relu')(env)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        dense3 = Dense(self.fc2_dims, activation='relu')(dense2)
        dense4 = Dense(self.fc2_dims, activation='relu')(dense3)
        probs = Dense(self.n_actions, activation='softmax')(dense4)
        '''
        '''
        6 layer dense relu sigmoid
         #input layer
        advantages = Input(name="advantage",shape=[1])
        env1d = Input(name="state",shape=(self.input_dims))
        env = Flatten()(env1d)
        #1st layer
        denseRelu1 = Dense(128, activation='relu')(env)
        dropout1 = Dropout(.3, input_shape=(128,))(denseRelu1)
        #2nd layer
        denseSigmoid2 = Dense(128, activation=activations.sigmoid)(dropout1)
        dropout2 = Dropout(.3, input_shape=(128,))(denseSigmoid2)
        #3rd layer
        denseRelu3 = Dense(128, activation='relu')(dropout2)
        dropout3 = Dropout(.3, input_shape=(128,))(denseRelu3)
        #4th layer
        denseSigmoid4 = Dense(64, activation=activations.sigmoid)(dropout3)
        dropout4 = Dropout(.3, input_shape=(64,))(denseSigmoid4)
        #5th layer
        denseRelu5 = Dense(32, activation='relu')(dropout4)
        dropout5 = Dropout(.3, input_shape=(64,))(denseRelu5)
        #6th layer
        denseSigmoid6 = Dense(8, activation=activations.sigmoid)(dropout5)
        #output layer
        probs = Dense(self.n_actions, activation='softmax')(denseSigmoid6)
        '''
        '''
        7 layer dense sigmoid relu sigmoid 
        '''
        #input layer
        compass = Input(name="compass",shape=[3])
        env1d = Input(name="state",shape=(self.input_dims))
        env = Flatten()(env1d)
        com_f = Flatten()(compass)
        
        # combine the output of the two branches
        state_reward = concatenate([env, com_f])
        
        #1st layer
        denseSigmoid1 = Dense(64, activation=activations.sigmoid)(state_reward)
        dropout1 = Dropout(.3, input_shape=(64,))(denseSigmoid1)
        #2nd layer
        denseRelu2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(.3, input_shape=(64,))(denseRelu2)
        #3rd layer
        denseSigmoid3 = Dense(64, activation=activations.sigmoid)(dropout2)
        dropout3 = Dropout(.3, input_shape=(64,))(denseSigmoid3)
        #4th layer
        denseRelu4 = Dense(32, activation='relu')(dropout3)
        dropout4 = Dropout(.3, input_shape=(32,))(denseRelu4)
        #5th layer
        denseSigmoid5 = Dense(16, activation=activations.sigmoid)(dropout4)
        dropout5 = Dropout(.3, input_shape=(16,))(denseSigmoid5)
        #6th layer
        denseRelu6 = Dense(16, activation='relu')(dropout5)
        dropout6 = Dropout(.3, input_shape=(16,))(denseRelu6)
        #7th layer
        denseSigmoid7 = Dense(8, activation=activations.sigmoid)(dropout6)
        #output layer
        probs = Dense(self.n_actions, activation='softmax')(denseSigmoid7)
       
        
        advantages = Input(name="advantage",shape=[1])
        def custom_loss(y_pred, y_true):
            out = K.clip(y_pred, 1e-8,  1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*advantages)
        
        policy = Model(inputs=[env1d,advantages, compass], outputs=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)
        
        self.predict = Model(inputs=[env1d,compass], outputs=[probs])
        
        return policy, self.predict
    
    def choose_action(self, observation,compass_h):
        state = observation[np.newaxis, :]
        compass = compass_h
        #norm = np.linalg.norm(state)
        #norm_state = state/norm
        
        probabilities = self.predict.predict(state,compass)
        action = np.random.choice(self.action_space, p=probabilities)
        
        return action
    
    
    def store_transition(self, observation, compass_h, action, reward):
        state = observation
        compass = compass_h
        self.action_memory.append(action)
        self.state_memory.append(state)
        self.compass_memory.append(compass)
        self.reward_memory.append(reward)
        
    #find position around agent funtion    
       
        
    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        compass_memory = np.array(self.compass_memory)
        
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *= self.gamma
                
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std
        
        cost = self.policy.train_on_batch([state_memory ,self.G, compass_memory], action_memory)
        
        self.state_memory = []
        self.compass_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    #return cost funtion
    
    def ep_graph(self,steps_arr,score_arr,action_arr,ep):
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
        
    def save_model(self):
        self.policy.save(self.model_file_policy)
        self.predict.save(self.model_file_predict)
    
    def load_m(self):
        advantages = self.G
        policy = None
        predict = None
        
        def custom_loss(y_pred, y_true):
            out = K.clip(y_pred, 1e-8,  1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*advantages)
        
        #load weights
        if (self.model_file_policy != None and self.model_file_predict != None and self.load == True):
            policy = load_model(self.model_file_policy, custom_objects={'custom_loss': custom_loss})
            predict = load_model(self.model_file_predict)
        else:
            policy, predict = self.build_policy_network()
        
        #get_custom_objects().update({'my_custom_func': my_custom_func})
        
        return policy, predict
    