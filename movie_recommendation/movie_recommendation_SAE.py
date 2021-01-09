# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:01:53 2021

@author: Sagnik_laptop
"""
# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Get the Dataset
movies = pd.read_csv('ml-1m/movies.dat', 
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')

users = pd.read_csv('ml-1m/users.dat', 
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')

ratings = movies = pd.read_csv('ml-1m/ratings.dat', 
                     sep = '::',
                     header = None,
                     engine = 'python',
                     encoding = 'latin-1')

# only using 100k user to train
# col 1: user id
# col 2: movie id
# col 3: user ratings

train_set = pd.read_csv('ml-100k/u1.base', sep = '\t', header = None)
train_set = np.array(train_set)

test_set = pd.read_csv('ml-100k/u1.test', sep = '\t', header = None)
test_set = np.array(test_set)

nb_users = int(max(max(train_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(train_set[:, 1]), max(test_set[:, 1])))

# Convert data into matrix where:
# each row is a user, each colum is a movie
# each cell (u, i) is rating user u, gave to movie i
def convert_data(data):
    conv_data = []
    for u_id in range(1, nb_users+1):
        # get all the movies that the user has watched
        movie_id = data[:, 1][u_id == data[:, 0]]
        # get all the ratings for those movies
        ratings = data[:, 2][u_id == data[:, 0]]
        # set the ratings in coresponding movie columns
        user_ratings = np.zeros(nb_movies)
        user_ratings[movie_id -1] = ratings
        conv_data.append(list(user_ratings))
    return conv_data

train_set = convert_data(train_set)
test_set = convert_data(test_set)

#covert to torch tensors
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# Create architecture of Stacked Auto Encoder
class SAE(nn.Module):
    def __init__(self, ):
        # call the init fuction of nn.Module class
        super(SAE, self).__init__()
        # connect encoding layer to input layer
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        # decoding
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        #encode x
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #decode x
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # get the constructed output
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training SAE
nb_epoch = 200
for e in range (1, nb_epoch +1):
    train_loss = 0
    s = 0.
    for u_id in range(nb_users):
        # create a batch of size 0
        x_in = Variable(train_set[u_id]).unsqueeze(0)
        target = x_in.clone()
        
        if torch.sum(target.data > 0) > 0:
            x_pred = sae.forward(x_in)
            target.require_grad = False # don't compute grad w.r.t target
            # predictions based on movies that were not watch don't count
            x_pred[target == 0] = 0
            loss = criterion(x_pred, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) 
                                             + 1e-10) # math safety
            loss.backward() # decide direction for updating weights
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step() # decide the factor by which weight is updated
    print('epoch: '+str(e)+' Train loss MSE: '+str(train_loss/s))
    
# Test SAE
test_loss = 0
s = 0.
for u_id in range(nb_users):
    x_in =  Variable(train_set[u_id]).unsqueeze(0)
    target =  Variable(test_set[u_id]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        x_pred = sae.forward(x_in)
        target.require_grad = False
        x_pred[target == 0] = 0
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) 
                                             + 1e-10)
        loss = criterion(x_pred, target)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss MSE: '+str(test_loss/s)) # -> 0.96 
# we're off by less than 1 star!!