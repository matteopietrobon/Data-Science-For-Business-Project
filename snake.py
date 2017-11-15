# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:52:28 2017

@author: utente
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

class Snake(object):
    def __init__(self, length, position):
        self.length = length
        self.position = position
        
    
        
        
class Game(object):
    def __init__(self, shape):
        self.status = 'playing'
        self.height = shape[0]
        self.width = shape[1]
        self.grid = np.zeros(shape)
        self.display = False
        self.plot_grid = np.zeros([self.height+2,self.width+2])
        self.plot_grid[1:self.height+1,1:self.width+1] = np.ones(shape)
        
        # Spawn the Snake at the center of the grid
        self.snake = Snake(2,np.array([[np.floor(self.height/2),np.round(self.width/2)],
                               [np.floor(self.height/2),np.round(self.width/2)-1]]).astype(int))
        for pos in self.snake.position:
            self.grid[pos[0],pos[1]] = 1 # update the grid
            
        self.apple = self.spawn_apple()
        
    def victory(self):
        self.status = 'won'
        print('GAME OVER: you won!')
        
    def loss(self):
        self.status = 'lost'
#        print('GAME OVER: you lost!') # commenting because in the training it is quite frequent
        
    def check_available(self,guess):
        # Check that the selected cell is at least 1 far from the snake's body
        return np.max(self.grid[
                np.max([0,guess[0]-1]):np.min([self.height,guess[0]+2]),
                np.max([0,guess[1]-1]):np.min([self.width,guess[1]+2])
                ]) == 0
        
    def spawn_apple(self):
        # Greedy approach: try one cell and check if it works
        n_tried = 0
        while n_tried < 10:
            n_tried += 1
            guess = np.random.rand(2)
            guess = np.array([np.floor(guess[0]*self.height),np.floor(guess[1]*self.width)]).astype(int)
            if self.check_available(guess):
                self.grid[guess[0],guess[1]] = -1
                return guess
        
        # Find all available cells and choose one, if none you won
        feasible = []
        for row in range(self.height):
            for col in range(self.width):
                if self.check_available([row,col]):
                    feasible.append([[row,col]])
        
        if len(feasible)>0:
            guess2 = np.random.rand(1)
            guess2 = int(np.floor(guess[0]*len(feasible)))
            self.grid[feasible[guess2]] = -1
            return feasible[guess2]
        else:
            self.victory()
            return None
        
    def get_movement(self,direction):
        # getting the direction of movement:
        # -1: left
        #  0: straight
        # +1: right
        current_pos = np.array(self.snake.position[0])
        body_dir = np.array(self.snake.position[1])
        straight = current_pos-body_dir
        
        if direction == 0:
            movement = straight
        elif direction == 1:
            movement = np.array([straight[1],-straight[0]])
        elif direction == -1:
            movement = np.array([-straight[1],straight[0]])
        else:
            raise ValueError('Direction not valid!!')
        return (current_pos+movement).astype(int)
        
    def move(self,direction):
        next_pos = self.get_movement(direction)
        
        # if you go into a wall or your body you lose
        if (next_pos[0]<0)or(next_pos[0]>=self.height)or(next_pos[1]<0)or(next_pos[1]>=self.width):
            self.loss()
        elif self.grid[next_pos[0],next_pos[1]] == 1:
            self.loss()
        # next position is empty, update the snake
        elif self.grid[next_pos[0],next_pos[1]] == 0:
            self.grid[next_pos[0],next_pos[1]] = 1
            self.grid[self.snake.position[0][0],self.snake.position[0][1]] = 1
            self.grid[self.snake.position[-1][0],self.snake.position[-1][1]] = 0
            self.snake.position = np.append([next_pos],self.snake.position[:-1],axis=0)
        # eat the apple
        elif self.grid[next_pos[0],next_pos[1]] == -1:
            self.grid[next_pos[0],next_pos[1]] = 1
            self.grid[self.snake.position[0][0],self.snake.position[0][1]] = 1
            self.snake.position = np.append([next_pos],self.snake.position,axis=0)
            self.snake.length += 1
            self.apple = self.spawn_apple()
        else:
            raise ValueError('Error while moving')
        
        if self.display:
            self.display_plot()
                   
    def display_plot(self):
        current_pos = np.array(self.snake.position[0])
        body_dir = np.array(self.snake.position[1])
        head_dir = (current_pos-body_dir).astype(int)
        if (head_dir == [1,0]).all():
            head='v'
        elif (head_dir == [-1,0]).all():
            head='^'
        elif (head_dir == [0,1]).all():
            head='>'
        if (head_dir == [0,-1]).all():
            head='<'
        
        if not self.display:
            self.display = True
            plt.figure()
            plt.imshow(self.plot_grid,cmap=plt.cm.gray)
            size_body = np.ceil(200/np.max([self.height, self.width]))
            size_marker = np.ceil(250/np.max([self.height, self.width]))
            if self.apple is None:
                self.plot_apple, = plt.plot([],[],color='r',markersize=size_marker,marker='o',zorder=2)
            else:
                self.plot_apple, = plt.plot(self.apple[1]+1,self.apple[0]+1,marker='o',
                                            color='r',markersize=size_marker,zorder=3)
            self.plot_head, = plt.plot(self.snake.position[0][1]+1,self.snake.position[0][0]+1,
                        color='g',marker=head,markersize=size_marker,markeredgecolor='k',zorder=3)
            self.plot_body, = plt.plot(np.array(self.snake.position).T[1]+1,
                                       np.array(self.snake.position).T[0]+1,
                                       color='g',marker='None',linewidth=size_body,zorder=1)
            plt.axis('off')
            plt.draw()
        else:
            if self.apple is not None:
                self.plot_apple.set_xdata(self.apple[1]+1)
                self.plot_apple.set_ydata(self.apple[0]+1)
            else:
                self.plot_apple.set_xdata([])
                self.plot_apple.set_ydata([])
            self.plot_head.set_marker(head)
            self.plot_head.set_xdata(self.snake.position[0][1]+1)
            self.plot_head.set_ydata(self.snake.position[0][0]+1)
            self.plot_body.set_xdata(np.array(self.snake.position).T[1]+1)
            self.plot_body.set_ydata(np.array(self.snake.position).T[0]+1)
            plt.draw()

        
class Neural_Net(object):
    def __init__(self,game,weights):
        self.n_layers = 1 #TODO: implement automatic weight for more layers
        self.layer_size = 2
        # inputs are direction of the apple (2 inputs) and state of the grid around the head:
        #
        #       * * *       H = head
        #       * H *       B = body
        #         B        
        self.n_inputs = 7 
        #  1st output: straight or turn
        #  2nd output: if turn, left or right
        self.n_outputs = 2
        
        self.game = game
        self.number_moves = 0
        self.weigths1 = np.array(
                [weights[counter*self.n_inputs:(counter+1)*self.n_inputs] for counter in range(self.layer_size)])
        self.weigthsout = np.array([weights[-2*self.layer_size:-1*self.layer_size],weights[-1*self.layer_size:]])
        
        self.grid = np.ones([self.game.height+2,self.game.width+2])
        
        
    def get_inputs(self):
        # return vector of the inputs
        current_pos = np.array(self.game.snake.position[0])+1
        body_pos = np.array(self.game.snake.position[1])+1
        apple_pos = np.array(self.game.apple)+1
        head_dir = current_pos-body_pos
        apple_dir = apple_pos-current_pos
        apple_dir = apple_dir/np.linalg.norm(apple_dir)
        self.grid[1:self.game.height+1,1:self.game.width+1] = self.game.grid
        
        movements = np.array([current_pos+head_dir,
                              current_pos+np.array([head_dir[1],-head_dir[0]]),
                              current_pos+np.array([-head_dir[1],head_dir[0]]),
                              current_pos+head_dir+np.array([head_dir[1],-head_dir[0]]),
                              current_pos+head_dir+np.array([-head_dir[1],head_dir[0]])]).astype(int)
        
        apple_data = np.array([
                np.dot(head_dir,apple_dir),
                head_dir[0]*apple_dir[1]-head_dir[1]*apple_dir[0]])
        return np.append(apple_data,self.grid[movements[:,0],movements[:,1]])
    
    def compute(self):
        # from inputs get outputs
        input_data = self.get_inputs()
        output = np.tanh(np.dot(self.weigthsout,np.tanh(np.dot(self.weigths1,input_data))))
        if output[0] > 0:
            return 0
        else:
            return np.sign(output[1])
    
    def get_score(self):
        #☺ score used for the learning
        self.score = min(100,self.number_moves)+\
                    (self.number_moves>50)*1000*self.game.snake.length/np.sqrt(self.number_moves)
        if self.game.status == 'won':
            self.score += 1000
        if self.game.status == 'lost':
            self.score -= 500
        return self.score

    def play(self,max_moves):
        # play until the game doesn't end or you don't make too many moves
        while self.game.status=='playing' and self.number_moves<max_moves:
            self.number_moves += 1
            self.game.move(np.round(self.compute()))
        return self.get_score()
        
    def play_with_display(self,max_moves):
        time.sleep(3)
        while self.game.status=='playing' and self.number_moves<max_moves:
            self.number_moves += 1
            self.game.move(np.round(self.compute()))
            plt.pause(0.1)
        return self.get_score()
        
     
        
n_gamers = 100
grid_size=[20,20]
epochs = 15
moves = 50
n_weights= 30

#♣ generate initial weights randomly
list_weights = np.random.randn(n_gamers,n_weights)
new_weights = np.zeros([n_gamers,n_weights])
list_scores = np.zeros([n_gamers,epochs])
species = np.zeros([n_gamers,epochs+1])
species[:,0] = np.arange(n_gamers)
    

for generation in range(epochs):
    print('Generation: '+str(generation))
    # get the score with every set of rules
    for counter,weights in enumerate(list_weights):
        game = Game(grid_size)
        net = Neural_Net(game,weights)
        list_scores[counter,generation] = net.play((generation+3)*moves)
    
    # compute the best combination, discard the worst and modify slightly the
    # best to replace them
    print('Best Score: '+str(max(list_scores[:,generation])/(generation+3)/moves))
    best_performers=list_scores[:,generation].argsort()[-int(n_gamers/2):][::-1]
    species[:,generation+1] = np.append(species[best_performers,generation],species[best_performers,generation])
    new_noise = np.random.randn(int(n_gamers/2),n_weights)*(0.5/(generation+1))
    new_weights[:int(n_gamers/2),:] = list_weights[best_performers,:]+new_noise
    new_weights[int(n_gamers/2):,:] = list_weights[best_performers,:]
    list_weights[:,:] = new_weights[:,:]

# see how many species survive
species_by_gen = pd.DataFrame(np.sum(species==0,axis=0)).T
for i in range(1,n_gamers):
    species_by_gen=species_by_gen.append(pd.DataFrame(np.sum(species==i,axis=0)).T,ignore_index=True)
    

column_list = ['Generation '+str(i) for i in range(epochs+1)]
rename_dict = {}
for current_col,real_col in zip(species_by_gen.columns,column_list):
    rename_dict[str(current_col)]=real_col
species_by_gen.rename(index=str, columns=rename_dict)   

species_by_gen.T.plot(kind="bar", stacked=True)
    
plt.figure()
plt.plot(np.max(list_scores/moves,axis=0)/(np.arange(epochs)+3))
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.show()

#%%

# display the game of the best performer
game = Game(grid_size)
game.display_plot()
net = Neural_Net(game,list_weights[best_performers[0],:])
net.play_with_display(100000)

#%%
