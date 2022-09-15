# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 06:42:03 2019

@author: Amit Elbaz
"""

from Robot import Robot
from World import World
from matplotlib import pyplot as plt
import numpy as np

'''
================ Added code for the robot class =================
    # Question b with seed to find true path:
    def move_seed(self,u1,u2):
        # u1 is angle adding
        # u2 is forward velocity
        
        # Real input with noise:
        np.random.seed(0) # Seeding
        u1_true = u1 + norm.rvs(scale = self.turn_noise)
        u2_true = u2 + norm.rvs(scale = self.forward_noise)
        
        # Updating:
        self.theta = self.theta + u1_true
        self.x = self.x + u2_true * np.cos(self.theta)
        self.y = self.y + u2_true * np.sin(self.theta)
        
        # Cycilic world:
        if self.theta > np.pi*2:
            self.theta -= np.pi*2
        elif self.theta < 0:
            self.theta += np.pi*2
        if self.x > 100:
            self.x -= 100
        elif self.x < 0:
            self.x += 100
        if self.y > 100:
            self.y -= 100
        elif self.y < 0:
            self.y += 100
    
    # Question b without seed for the particle filter:
    def move(self,u1,u2):
        # u1 is angle adding
        # u2 is forward movement
        
        # Real input with noise:
        u1_true = u1 + norm.rvs(scale = self.turn_noise)
        u2_true = u2 + norm.rvs(scale = self.forward_noise)
        
        # Updating:
        self.theta = self.theta + u1_true
        self.x = self.x + u2_true * np.cos(self.theta)
        self.y = self.y + u2_true * np.sin(self.theta)
        
        # Cycilic world:
        if self.theta > np.pi*2:
            self.theta -= np.pi*2
        elif self.theta < 0:
            self.theta += np.pi*2
            
        if self.x > 100:
            self.x -= 100
        elif self.x < 0:
            self.x += 100
        if self.y > 100:
            self.y -= 100
        elif self.y < 0:
            self.y += 100
    
    
    # Question c:
    def sense(self,x_mean,y_mean,theta_mean):
        
        my_world = World()
        Landmarks = my_world.landmarks
        # Ranges:
        Ranges = np.zeros(6)
        Bearings = np.zeros(6)
        for index , L in enumerate(Landmarks):
            
            L = np.array(L)
            Ranges[index] = np.linalg.norm([x_mean,y_mean]-L)\
                          + norm.rvs(scale = self.sense_noise_range)
            
            Bearings[index] = np.arctan2(y_mean-L[1] , x_mean-L[0]) - \
                    theta_mean + norm.rvs(scale = self.sense_noise_bearing)
            
        return Ranges,Bearings
    
    # Question d:
    def measurement_probability(self , x_mean , y_mean , theta_mean , Ranges , Bearings):
        
        
        my_world = World()
        Landmarks = my_world.landmarks
        
        # Ranges:
        Ranges_mean = np.zeros(6)
        Bearings_mean = np.zeros(6)
        for index , L in enumerate(Landmarks):
            L = np.array(L)
            Ranges_mean[index] = np.linalg.norm([self.x,self.y]-L)
            Bearings_mean[index] = np.arctan2(self.y-L[1] , self.x-L[0]) - self.theta
        
        q = 1
        for R_mean , B_mean , R , B in zip (Ranges_mean,Bearings_mean,
                                            Ranges,Bearings):
            q *= norm.pdf(R,R_mean,self.sense_noise_range)\
                * norm.pdf(B,B_mean,self.sense_noise_bearing)
        
        return q
'''

def particle_filter(X_last,u,X_real):
    global N # 1000 particles.
    
    # Initializing location and weight arrays:
    W_n = np.zeros(N)
    robot_location = np.zeros((3,N))
    # Initializing Robot classes:
    myrobot = Robot()
    real_robot = Robot()
    # Setting the noises:
    myrobot.set_noise(new_forward_noise=6 , new_turn_noise=0.1 ,
                      new_sense_noise_range=5 ,  new_sense_noise_bearing=0.3)
    
    real_robot.set_noise(new_forward_noise=6 , new_turn_noise=0.1 ,
                      new_sense_noise_range=5 ,  new_sense_noise_bearing=0.3)
    
    # Getting the Range and Bearing measurements from the sensor:
    Range , Bearing =real_robot.sense(X_real[0],X_real[1],X_real[2])
    # Motion,plot and weight step:
    for index in range(0,N):
        myrobot.set(X_last[0,index] , X_last[1,index] , X_last[2,index])
        myrobot.move(u[0],u[1])
        myrobot.plot(mycolor = 'black' , style = 'particle' ,
                     show = True , markersize=2)
        
        W_n[index] = myrobot.measurement_probability(X_real[0],X_real[1],
                                                     X_real[2],Range,Bearing)
        robot_location[:,index]=myrobot.get_pose()
        
    # Resampling step:
    W_n = wight_normalize(W_n)
    New_X = resample(W_n , robot_location.T)
    
    # Ploting step:
    for x,y,theta in New_X.T:
        myrobot.set(x , y , theta)
        myrobot.plot(mycolor = 'grey' , style = 'particle' ,
                     show = True , markersize=1.2)
    
    # Robot mean plot:
    x_mean , y_mean , theta_mean = weighted_mean(New_X)
    myrobot.set(x_mean , y_mean , theta_mean)
    myrobot.plot(show=True)
    
    return New_X , x_mean , y_mean

# Normalizing the weights:
def wight_normalize(old_weights):
    global N
    old_weights = np.array(old_weights)
    new_weights = old_weights/np.sum(old_weights)
    return new_weights

def weighted_mean(New_X):
    return np.mean(New_X,axis = 1)

# Updating the particles with the weights:
def resample(weights , X):
    global N
    New_X = np.zeros((3,N))
    X = np.array(X).T
    X_indexes = np.int16(np.linspace(0,999,1000))
    # resample indexes:
    weight_index = np.random.choice(X_indexes, N, p=weights)
    for index , weight in enumerate(weight_index):
        New_X[:,index] = X[:,weight_index[index]]
    
    return np.array(New_X)
    

def question_a():
       # Initializing poses:
       x1,y1,theta1 = np.array([45,45,0])
       x2,y2,theta2 = np.array([50,60,0.5*np.pi])
       x3,y3,theta3 = np.array([70,30,0.75*np.pi])
       
       myworld = World()
       myworld.plot(show=True)
       myrobot = Robot()
       # First Robot:
       myrobot.set(x1,y1,theta1)
       myrobot.plot(show = True)
       # Second robot:
       myrobot.set(x2,y2,theta2)
       myrobot.plot(show = True)
       # Third robot:
       myrobot.set(x3,y3,theta3)
       myrobot.plot(show = True)
       

def question_e():
    global U1,U2
    
    myworld = World()
    myworld.plot(show=True)
    
    x_input = [10]
    y_input = [15]
    
    # Robot class:
    myrobot = Robot()
    # noises:
    myrobot.set_noise(0,0,0,0)
    # Initial pose:
    myrobot.set(10,15,0)
    myrobot.plot(show = True)
    for u1,u2 in zip (U1,U2):
        myrobot.move(u1,u2)
        myrobot.plot(show = True)
        # To plot the path:
        x_input.append(myrobot.x)
        y_input.append(myrobot.y)
    
    plt.plot(x_input, y_input, linestyle = 'dotted', label = 'Input',
             color='red', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    plt.show()
    
    
def question_f():
    global U1,U2
    
    myworld = World()
    # Robot class:
    myrobot = Robot()
    myworld.plot(show=True)
    
    x_input = [10]
    y_input = [15]
    myrobot.set_noise(0,0,0,0)
    # Initial pose:
    myrobot.set(10,15,0)
    myrobot.plot(show = True)
    for u1,u2 in zip (U1,U2):
        myrobot.move(u1,u2)
        # To plot the path:
        x_input.append(myrobot.x)
        y_input.append(myrobot.y)
    
    plt.plot(x_input, y_input, linestyle = 'dotted', label = 'Input',
             color='red', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    
    # First location for path:
    x_true = [10]
    y_true = [15]
    theta_true = [0]
    
    # noises:
    myrobot.set_noise( 6 , 0.1 , 5 , 0.3)
    # Initial pose:
    myrobot.set(10,15,0)
    myrobot.plot(show = True)
    for u1,u2 in zip (U1,U2):
        myrobot.move_seed(u1,u2)
        myrobot.plot(show = True)
        # Adding all the locations:
        x_true.append(myrobot.x)
        y_true.append(myrobot.y)
        theta_true.append(myrobot.theta)
    
    plt.plot(x_true, y_true, linestyle = '-', label = 'True path',
             color='green', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    
    plt.show()
    # The seeding:
    return x_true , y_true , theta_true
    
    
def question_g(x_true , y_true , theta_true):
    
    
    global N , U1 , U2
    myworld = World()
    myworld.plot(show=True)
    
    X = np.zeros((3,N))
    x = np.array([10,15,0])
    x_mean_est = [10]
    y_mean_est = [15]
    myrobot = Robot()
    
    # Initial pose:
    myrobot.set(10,15,0)
    myrobot.plot(show = True)
    
    for i in range(0,N):
        X[:,i] = x
    
    # Particle filter section:
    print ('Please wait, particle filter is in action...')
    for u1,u2,x_real in zip(U1,U2,
                            np.array([x_true[1:],y_true[1:],
                                      theta_true[1:]]).T):
        X , x_mean , y_mean = particle_filter(X,[u1,u2],x_real)
        x_mean_est.append(x_mean)
        y_mean_est.append(y_mean)
        print (f'plotted {u1,u2} action')
    
    
    # Input data section:
    x_input = [10]
    y_input = [15]
    myrobot.set_noise(0,0,0,0)
    # Initial pose:
    myrobot.set(10,15,0)
    myrobot.plot(show = True)
    for u1,u2 in zip (U1,U2):
        myrobot.move(u1,u2)
        # To plot the path:
        x_input.append(myrobot.x)
        y_input.append(myrobot.y)
    
    
    # Plotting the pathes section:
    plt.plot(x_input, y_input, linestyle = 'dotted', label = 'Input',
             color='red', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    
    plt.plot(x_true, y_true, linestyle = '-', label = 'True path',
             color='green', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    
    plt.plot(x_mean_est, y_mean_est, linestyle = '-.',
             label = 'Particle filter',color='black', markersize = 1)
    plt.legend(loc = 1 , fontsize = 'xx-small')
    
    plt.show()
    
        
# Main:
Menu_msg = 'Insert a letter between a,e,f,g for the relevent question or q to quit: '
Error_msg = 'Inerted invalid letter. Try again'
No_path_msg = 'True path has yet to be implemented. Choose option "f" first to implement it.'
# Actions:
global U1,U2
U1 = [0,np.pi/3,np.pi/4,np.pi/4,np.pi/4]
U2 = [60,30,30,20,40]
# Particles:
global N

x = []
ans = 'b'

while ans != 'q':
       N = 1000
       ans = input(Menu_msg)
       if ans == 'a':
              question_a()
       elif ans == 'e':
              question_e()
       elif ans == 'f':
              x , y , theta = np.array(question_f())
       elif ans == 'g':
              if len(x) == 0:
                  print (No_path_msg)
                  continue
              question_g(x,y,theta)
       elif ans == 'q':
              print('Goodbye')
       else:
              print(Error_msg)


