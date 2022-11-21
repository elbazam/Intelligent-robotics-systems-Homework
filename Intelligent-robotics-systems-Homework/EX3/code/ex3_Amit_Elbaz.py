# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:25:40 2020

@author: Amit Elbaz
"""
import numpy as np
from World import World
import matplotlib.pyplot as plt


def _AN(s,n_s):
    '''North action.
    Return 1 if reached the new state or 0 if not '''
    north = s - 1
    if north % 4 == 0 or north in [6,7]:
        north = s
    return 1 if north == n_s else 0
    

def _AE(s,n_s):
    '''East action.
    Return 1 if reached the new state or 0 if not '''
    east = s + 4
    if east > 16 or east in [6,7]:
        east = s
    return 1 if east == n_s else 0


def _AS(s,n_s):
    '''South action.
    Return 1 if reached the new state or 0 if not '''
    south = s + 1
    if s % 4 == 0 or south in [6,7]:
        south = s
    return 1 if south == n_s else 0
    
    
def _AW(s,n_s):
    '''West action.
    Return 1 if reached the new state or 0 if not '''
    west = s - 4
    if west < 1 or west in [6,7]:
        west = s
    return 1 if west == n_s else 0
    

class mdp():
    def __init__(self,r=-0.04,gamma=1):
        """Initialize reward and gamma"""
        self.r = r
        self.gamma = gamma
        # The reward function, question a.
        self.R_s = np.array([r , r , r , r,
                             r , 0 , 0 , r,
                             r , r , r , r,
                             1 , r , -1, r])
    
        self.A = [1,2,3,4] # All the actions.
        self.States = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # All the states.
        self.NMStates = [6,7,13,15] # Robot cannot move from those states.
    
    def _Transition_model(self,n_s,s,a):
        '''Return the right cell in the transision matrix.
        The answer for question a part 2'''
        if s in self.NMStates: # Terminal state or unvalid place
            return 0
        if a == 1: # North action
            return 0.8*_AN(s,n_s) + 0.1*(_AE(s,n_s) + _AW(s,n_s))
        elif a == 2: # East action
            return 0.8*_AE(s,n_s) + 0.1*(_AN(s,n_s) + _AS(s,n_s))
        elif a == 3: # South action
            return 0.8*_AS(s,n_s) + 0.1*(_AE(s,n_s) + _AW(s,n_s))
        else: # West action
            return 0.8*_AW(s,n_s) + 0.1*(_AN(s,n_s) + _AS(s,n_s))
    
    def value_iteration(self , threshold = 10**(-4)):
        """value iteration, return Value vector and Policy vector"""
        V = np.zeros(16)
        theta = 1 # Initialize theta
        while theta > threshold:
            theta = 0
            ''' The value evaluation part.'''
            for s in self.States:
                V_a = np.zeros(4)
                v = V[s-1]
                for a in self.A:
                    for new_s in self.States:
                        P = mdp._Transition_model(self,new_s,s,a)
                        V_a[a-1] += P * (self.gamma * V[new_s-1])
                        
                V[s-1] = np.max(V_a) + self.R_s[s-1]
                theta = np.max([theta,np.absolute(v-V[s-1])])
                if theta < threshold:
                    break
                
        Policy = np.zeros(16)
        '''Policy evaluation part '''
        for s in self.States:
                if s in self.NMStates:
                    continue
                V_a = np.zeros(4)
                for a in self.A:
                    for new_s in self.States:
                        P = mdp._Transition_model(self,new_s,s,a)
                        if P == 0:
                            continue
                        V_a[a-1] += P * (self.gamma*V[new_s-1])
                        
                Policy[s-1] = np.argmax(V_a + self.R_s[s-1]) + 1
        return V , Policy
    
    
    def _Policy_evaluation(self,Policy,threshold = 10**(-4)):
        '''Return values for a given policy'''
        V = np.zeros(16)
        theta = 1
        while theta > threshold:
            theta = 0
            for s in self.States:
                v = V[s-1]
                V_a = np.zeros(4)
                a = Policy[s-1]
                for new_s in self.States:
                    P = mdp._Transition_model(self,new_s,s,a)
                    V_a[a-1] += P * (self.gamma*V[new_s-1])       
                V[s-1] =  np.sum(V_a) + self.R_s[s-1]
                theta = np.max([theta,np.absolute(v-V[s-1])])
                if theta < threshold:
                    break
        return V
    
    def _Policy_Improvement(self, V ):
        '''Return optimal policy for a given Values '''
        Q = np.zeros((16,4))
        for s in self.States:
            for a in self.A:
                for new_s in self.States:
                    P = mdp._Transition_model(self,new_s,s,a)
                    Q[s-1,a-1] += P * (self.gamma*V[new_s-1])
        Policy = np.zeros(16)
        for s in self.States:
            if s in self.NMStates:
                continue
            Policy[s-1] = np.int(np.argmax(Q[s-1,:]+self.R_s[s-1]) + 1)
        
        return Policy.astype(int)
    
    def _Policy_initialization(self):
        '''Return uniform random policy.'''
        Prob = [0.25,0.25,0.25,0.25]
        Policy = np.random.choice(self.A,16,Prob)
        return Policy
    
    def Policy_iteration(self,i = 6):
        w = World()
        Policy = mdp._Policy_initialization(self)
        while True:
            V = mdp._Policy_evaluation(self,Policy)
            '''Ploting part:'''
            plt.figure(i)
            w.plot_value(V)
            new_Policy = mdp._Policy_Improvement(self, V )
            plt.figure(i+1)
            w.plot_policy(new_Policy)
            i += 2
            '''If the same policy has been reached:'''
            if np.array_equal(Policy,new_Policy):
                return V , Policy
            '''In case it isn't:'''
            Policy = new_Policy


msg = 'Place number of question b~e or q to exit: '
error = 'Invalid input! Try again.'
w = World()
index = 0
while True:
    '''Menu, plot the question answer b~e.'''
    ans = input(msg)
    if ans not in ['b','c','d','e','q']:
        print(error)
        continue
    if ans == 'q':
        break
    elif ans == 'b':
        a_q = mdp()
        Value , Policy = a_q.value_iteration()
        plt.figure(0)
        w.plot_value(Value)
        plt.figure(1)
        w.plot_policy(Policy)
        index += 2
        break
    elif ans == 'c':
        a_q = mdp(gamma = 0.9)
        Value , Policy = a_q.value_iteration()
        plt.figure(2)
        w.plot_value(Value)
        plt.figure(3)
        w.plot_policy(Policy)
        index += 2
        break
    elif ans == 'd':
        a_q = mdp(r = -0.02)
        Value , Policy = a_q.value_iteration()
        plt.figure(4)
        w.plot_value(Value)
        plt.figure(5)
        w.plot_policy(Policy)
        index += 2
        break
    else:
        a_q = mdp(gamma = 0.9)
        Value , Policy = a_q.Policy_iteration()
        break



        
