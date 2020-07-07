# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:53:59 2020

@author: sefun
"""
import numpy as np
import matplotlib.pyplot as plt

class CGridWorld(object):
    def __init__(self,size, p, prefAction):       
            self.prefAction = prefAction
            self.actionSpace = {'U', 'D', 'L', 'R'}
            self.possibleActions = ['U', 'D', 'L', 'R']
            
            self.size = size
            self.p = p
            
            self.generate_random_map(size=5, p = self.p)
            self.setState()
            
    def onTree(state, action):
        #if grid value at center is equal to tree at center loction set on tree to true
        #set loction based off action
        if action == 'U':
            row = 0
            col = 1
        elif action == 'D':
            row = -1
            col = 1
        elif action == 'L':
            row = 1
            col = 0
        elif action == 'R':
            row = 1
            col = -1
            
        #if location matches tree 
        if state[row][col] == 'o':
            return True
        
        return False
    
    
    # DFS to check that it's a valid path.
    def is_valid(res, rsize, csize):
        #tracks a list of nodes from from (0,0) to end of graph goal
        frontier, discovered = [], set()
        frontier.append((0,0))
        #loops for all connected nodes till goal node
        while frontier:
            #row column position
            r, c = frontier.pop()
            #adds to examins and adds to discovered 
            if not (r,c) in discovered:
                discovered.add((r,c))
                #possible x ,y directions
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                #loops through each direction
                for x, y in directions:
                    #increasec row and colum postions exmined in x,y directions 
                    r_new = r + x
                    c_new = c + y
                    #examines node and added new nods to exmin to frontier
                    if r_new < 0 or r_new >= rsize or c_new < 0 or c_new >= csize:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] !='o'):
                        frontier.append((r_new, c_new))
        return False
    
   #generates start enviroment
    def generate_random_map(self, size = self.size+2, p):
        """Generates a random valid map (one that has a path from start to goal)
        :param size: size of each side of the grid
        :param p: probability that a tile is frozen
        """
        valid = False
        # loop tile valid generated
        while not valid:
            #generate random array of trres and gaps  bases of probabilty of polulation
            self.p = min(1, self.p)
            res = np.random.choice(['-', 'o'], (size, size), self.p=[self.p, 1-self.p])
            
            #set center value to agent
            res[2][2] = 'X'
            
            #set boarder edge to goal for valid travesal
            for i in range( self.size):
                res[0][i] = 'G'
                res[-1][i] = 'G'   
                res[i][0] = 'G'
                res[i][-1] = 'G'
             
            #check if valid
            valid = self.is_valid(res, size, size)
            
        #remove goal edge    
        res = np.delete(res, 0, 0)
        res = np.delete(res, -1, 0)
        res = np.delete(res, 0, 1)
        res = np.delete(res, -1, 1)
            
        return res 
    
    def setState(self, state = self.generate_random_map(size=5, p=0.6)):
        sState = state
        return sState
    
    #Generate new row of column of map
    def stateUpdate(self, state, size, action, p = self.p):
        """Generates a random valid map (one that has a path from start to goal)
        :param size: size of each side of the grid
        :param p: probability that a tile is frozen
        """
        valid = False
        
        #loops until valid genration
        while not valid:
            
            #decides col/ rows to removed based off actions taken and determins withether generatiig a row or colum in newstate
            if action == 'U':
                rowCol = 0
                direction = -1
                genRow = 1
            elif action == 'D':
                rowCol = 0
                direction = 0
                genRow = 1
            elif action == 'L':
                rowCol = 1
                direction = -1
                genRow = 0
            elif action == 'R':
                rowCol = 1
                direction = 0
                genRow = 0
            
            
            #sets new space genreation based of column or row being generated
            if genRow == 1:
                rowSize = 2
                colSize = self.size
            else:
                rowSize = self.size
                colSize = 2
            
            #removes last colm/row based of action
            newState= np.delete(state, direction, rowCol)
            
            #repositions agent X to center space
            if not onTree(newState, action):
                if action == 'U':
                    temp = newState[0][1]
                    newState[0][1] = newState[1][1]
                    newState[1][1] = temp
    
                elif action == 'D':
                    temp = newState[1][1]
                    newState[1][1] = newState[0][1]
                    newState[0][1] = temp
    
                elif action == 'R':
                    temp = newState[1][1]
                    newState[1][1] = newState[1][0]
                    newState[1][0] = temp
    
                elif action == 'L':
                    temp = newState[1][0]
                    newState[1][0] = newState[1][1]
                    newState[1][1] = temp
            else:
                return None
                
            
            #generates new statespace and obstical poulation based off pprobability
            p = min(1, p)
            newRCArr = np.random.choice(['-', 'o'], (rowSize, colSize), p=[p, 1-p])
            newRCStr = ["".join(x) for x in newRCArr]
        
            
            #adds new space and sets furthest row/col as goal for path validation DFS
            if action == 'U':
                newState = np.concatenate((newRCArr, newState), axis=0)
                for i in range(size):
                    newState[0][i] = 'G'
            elif action == 'D':
                newState = np.concatenate((newState, newRCArr), axis=0)
                for i in range(size):
                    newState[-1][i] = 'G'
            elif action == 'L':
                newState = np.concatenate((newRCArr, newState), axis=1)
                for i in range(size):
                    newState[i][0] = 'G'
            elif action == 'R':
                newState = np.concatenate((newState, newRCArr), axis=1)
                for i in range(size):
                    newState[i][-1] = 'G'
             
            
            #determins wither generated row/col is valid
            if genRow == 1:
                valid = is_valid(newState, rsize = 2+size, csize = self.size)
            elif genRow == 0:
                valid = is_valid(newState, rsize = self.size, csize = 2+size)
                
            
            
        #delete goal row/col
        if action == 'U':
            newState = np.delete(newState, 0, 0)
        elif action == 'D':
            newState = np.delete(newState, -1, 0)
        elif action == 'L':
            newState = np.delete(newState, 0, 1)
        elif action == 'R':
            newState = np.delete(newState, -1, 1)       
            
            
        #return ["".join(x) for x in newState]
        return newState
    
    
    
    def step(self, action):
        
        action = self.actionSpace[action]
        
        reward = 0 
    
        if not onTree(self.setState(), action):
            
            resultingState = stateUpdate(self.setState(), size=3, action = action, p=0.6)
            
            reward += 1
            
            if resultingState == None:
                reward += -100
              
            #prefred action function
            if self.actionSpace[action] == self.prefAction:
                reward += -1
            else:
                reward += -2
            
            self.setState(resultingState, action)
            return resultingState, reward, \
                   self.onTree(resultingState), None
                   
        else:
            
            if resultingState == None:
                reward += -100
              
            #prefred action function
            if self.actionSpace[action] == self.prefAction:
                reward += -1
            else:
                reward += -2
            
            return resultingState, reward, \
                   self.onTree(resultingState), None
        

    def reset(self):
        setState()
        return setState()
    
    
    def render():
        print('------------------------------------------')
        for row in setState():
            for col in row:
                if col == '-':
                    print('-', end='\t')
                elif col == 'X':
                    print('X', end='\t')
                elif col == 'o':
                    print('o', end='\t')
            print('\n')
        print('------------------------------------------')


    def actionSpaceSample(self):
            return np.random.choice(self.possibleActions)
        
        
        