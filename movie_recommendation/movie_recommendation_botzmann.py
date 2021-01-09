# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:46:49 2021

@author: Sagnik_laptop
"""

# Movie recommendation system using
# restricted boltzmann machine
# data src: https://grouplens.org/datasets/movielens/
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import dataset
movies = pd.read_csv('ml-1m/movies.dat', 
                     sep = '::', 
                     header = None, 
                     engine='python', 
                     encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', 
                     sep = '::', 
                     header = None, 
                     engine='python', 
                     encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', 
                     sep = '::', 
                     header = None, 
                     engine='python', 
                     encoding='latin-1')
# column 1: user
# column 2: movie id
# column 3: user ratings

# Prepare train and test set
# Use standard 80-20, split to train
train_set = pd.read_csv('ml-100k/u1.base', sep = '\t', header = None)
train_set = np.array(train_set)

test_set = pd.read_csv('ml-100k/u1.test', sep = '\t', header = None)
test_set = np.array(test_set)


# getting the number of users and movies
nb_user = int(max(max(train_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(train_set[:, 1]), max(test_set[:, 1])))

# Create nxn matrix where each cell (u,i) is a rating
# u is the user who watched the movie i 
# if user didn't watch, rating is 0 
# each movie is a feature, each row is an observation about a user
def convert(data):
    conv_data = []
    for u_id in range(1, nb_user+1):
        # the movies the user watched
        movies_id = data[:, 1][data[:, 0] == u_id]
        rating_id = data[:, 2][data[:, 0] == u_id]
        
        ratings = np.zeros(nb_movies)
        # repace with the watched movies with given rating
        ratings[movies_id - 1] = rating_id
        conv_data.append(list(ratings))
    return conv_data

train_set = convert(train_set)
test_set = convert(test_set)

# Covert data into Torch tesors (multi dim with single d-type based on pytorch)
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# Covert rating into binary ratings 1 (Liked) or 0 (Not liked)
# set unwatched movies as -1
train_set[train_set == 0] = -1
test_set[test_set == 0] = -1

# set ratings with below 3 as 0
#Note: or doesn't work pytorch
train_set[train_set == 1] = 0
test_set[test_set == 1] = 0

train_set[train_set == 2] = 0
test_set[test_set == 2] = 0

# set 3 or above as 1
train_set[train_set >= 3] = 1
test_set[test_set >= 3] = 1

# Create architecture of probabilistic botzmann model
# nv: # visible nodes
# nh; # hidden nodes
# a: bias for (h_i|v_i) 
# b: bias (v_i|h_i)
# functions: init(), sample_h(), sample_v(), train()
class RBM():
    def __init__(self, nv, nh):
         #creates matrix of size nh, nv with normal distribution
         # based on P(nv_i|nh_i)
        self.W = torch.randn(nh, nv) #(weight matrix P(v|h))
        self.a = torch.randn(1, nh) # (batch, number of bias)
        self.b = torch.randn(1, nv)
        
    # calculate activation prob of hidden node given the visible node
    # then return sample of activated neurons based on the probabilities
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        # apply bias to each element of x
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # calculate prob of visible node given hidden node
    # (reconstructing the input)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #train using contrastive divergence (k-step)
    def train(self, v0, vk, ph0, phk):
        #update weights
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        # update biases
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
       

# Create instance of class
# each movie is a input
nv = len(train_set[0])
nh = 100 # number of features we want to extract
batch_size = 100
rbm = RBM(nv, nh)

# Train the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for user_id in range(0, nb_user - batch_size, batch_size):
        vk = train_set[user_id : user_id + batch_size]
        v0 = train_set[user_id : user_id + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
  
# Testing the RBM
test_loss = 0
s = 0.
for user_id in range(0, nb_user):
    # predict for one user at a time
    # Input row from the train, to predict row in the test
    v = train_set[user_id : user_id + 1]
    vt = test_set[user_id : user_id + 1]
    if len(vt[vt>=0]) > 0:
        _,activated_nodes = rbm.sample_h(v)
        _,v = rbm.sample_v(activated_nodes)
        test_loss += torch.mean(torch.abs(v[vt >= 0] - vt[vt >= 0]))
        s += 1.
print('Test loss: '+str(test_loss/s))

            