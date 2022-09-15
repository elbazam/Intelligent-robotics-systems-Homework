# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:06:29 2019
@author: Amit Elbaz

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Kalman filter function
def KF(mu_prev,sigma_prev,
       ut,zt,Rt,Qt):
       
       # Initializing:
       dt = 0.1
       A = np.array([[1,0,0,dt,0,0],
                     [0,1,0,0,dt,0],
                     [0,0,1,0,0,dt],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]])
       
       B = np.array([[0.5*(dt**2),0,0],
                      [0,0.5*(dt**2),0],
                      [0,0,0.5*(dt**2)],
                      [dt,0,0],
                      [0,dt,0],
                      [0,0,dt]])
       
       C = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])
       
       # Prediction:
       mu_star = np.matmul(A , mu_prev) + np.matmul(B , ut)
       sigma_star = np.matmul(np.matmul(A,sigma_prev),A.T) + Rt
       
       # Kalman gain:
       Kt = np.matmul(np.matmul(sigma_star , C.T) ,
                      np.linalg.inv(np.matmul(np.matmul(C,sigma_star),C.T) + Qt))
       
       # Measurment update:
       mu = mu_star + np.matmul(Kt,zt - np.matmul(C , mu_star))
       sigma = np.matmul((np.eye(6) - np.matmul(Kt , C)) , sigma_star)
       
       return mu,sigma

def figure_1_maker(t, x, y, theta, vx, vy, omega):
       
       # create a figure 
       fig = plt.figure()
       
       # creating subplots:
       plt1 = fig.add_subplot(231) 
       plt2 = fig.add_subplot(232) 
       plt3 = fig.add_subplot(233) 
       plt4 = fig.add_subplot(234)
       plt5 = fig.add_subplot(235)
       plt6 = fig.add_subplot(236) 
       
       # 1st plot:
       plt1.plot(t,x,linewidth = 0.5)
       plt1.set_title('$x(t)$',size = 15)
       plt1.set_xlabel('$t [sec]$',size = 15)
       plt1.set_ylabel('$x [mm]$',size = 15)
       
       # 2nd plot:
       plt2.plot(t,y,linewidth = 0.5)
       plt2.set_title('$y (t)$',size = 15)
       plt2.set_xlabel('$t [sec]$',size = 15)
       plt2.set_ylabel('$y [mm]$',size = 15)
       
       # 3rd plot:
       plt3.plot(t,theta,linewidth = 0.5)
       plt3.set_title(r'$\theta (t)$',size = 15)
       plt3.set_xlabel('$t [sec]$',size = 15)
       plt3.set_ylabel(r'$\theta [^\circ]$',size = 15)
       
       # 4th plot:
       plt4.plot(t,vx,linewidth = 0.5)
       plt4.set_title('$v_x (t)$',size = 15)
       plt4.set_xlabel('$t [sec]$',size = 15)
       plt4.set_ylabel(r'$v_x [\frac{mm}{sec}$]',size = 15)
       
       # 5th plot:
       plt5.plot(t,vy,linewidth = 0.5)
       plt5.set_title('$v_y (t)$',size = 15)
       plt5.set_xlabel('$t [sec]$',size = 15)
       plt5.set_ylabel(r'$v_y \frac{mm}{sec}$',size = 15)
       
       # 6th plot:
       plt6.plot(t,omega,linewidth = 0.5)
       plt6.set_title('$\omega (t)$',size = 15)
       plt6.set_xlabel('$t [sec]$',size = 15)
       plt6.set_ylabel(r'$\omega \frac{\circ}{sec}$',size = 15)
       
       fig.subplots_adjust(hspace=1.2,wspace=0.85)
       
def figure_2_maker(x ,y, c,ans='a',
                   x_measurment=[] ,y_measurment=[] ,
                   cov = [],mu=[]):
       
       fig, ax = plt.subplots()
       if c == 'blue':
              l = 0.5
       else:
              l = 1
       plt.plot(x,y,label = "$xy-path$",
                color = c,linewidth = l)
       
       if len(y_measurment) != 0:
              plt.scatter(x_measurment,y_measurment,
                          label="$measurments$",color = 'green',
                          marker='+' , s=40)
       if ans == 'e':
              for i in range(0,5):
                     # Compute eigenvalues
                     # and associated eigenvectors
                     vals, vecs = np.linalg.eigh(cov[i,:,:])
                     
                     # Compute "tilt" of ellipse using first eigenvector
                     x_v, y_v = vecs[:, 0]
                     theta = np.degrees(np.arctan2(y_v, x_v))
                     # Eigenvalues give length
                     # of ellipse along each eigenvector
                     w, h = 2 * np.sqrt(vals)
                     ellipse = Ellipse((mu[i,:]), w, h,
                                       theta, edgecolor='red',
                                       fill = False)  
                     ax.add_artist(ellipse) 
              
       ax.set_title('$y (x)$',size = 15)
       ax.set_xlabel("$x [mm]$",size = 15)
       ax.set_ylabel("$y [mm]$",size = 15)
       ax.legend()

def Q_b():
       
       global ground_truth  , pose
       # Initializing:
       t = ground_truth[:,0]
       x = ground_truth[:,1]
       y = ground_truth[:,2]
       theta = ground_truth[:,3]
       vx = ground_truth[:,4]
       vy = ground_truth[:,5]
       omega = ground_truth[:,6]
       x_measurment = pose[:,1]
       y_measurment = pose[:,2]    
       # Send to plot
       figure_1_maker(t,x,y,theta,vx,vy,omega)
       figure_2_maker(x,y,"black",x_measurment=x_measurment,
                      y_measurment=y_measurment)
       
def Q_cd(ans):
       
       global controls , pose
       dt = 0.1
       mu = []
       sigma = []
       
       #Initializing:
       t = np.array(controls[:,0])
       z = np.array(pose[:,1:4])
       u = np.array(controls[:,1:4])
       
       # Initial state vector and cov:
       new_mu = np.array([0,0,0,0,0,0])
       new_sigma = np.diag([200**2,200**2,200**2,
                            300**2,300**2,400**2])
         
       # Calculating Rt:
       Rt_eps = np.array([[10*0.5*dt**2,10*0.5*dt**2,10*0.5*dt**2,
                           10*dt,10*dt,10*dt]]).T
       
       # Making Rt:
       Rt = np.matmul(Rt_eps,Rt_eps.T)
       
       if ans == 'c':
              Qt = np.diag([200**2,200**2,200**2])
       else:
              Qt = np.diag([10**2,10**2,10**2])
       
       # Start to make the mu (mean) and sigma (cov):
       mu.append(new_mu)
       sigma.append(new_sigma)
           
       for ii in range (1,t.shape[0]):
              new_mu , new_sigma = KF(new_mu,new_sigma,u[ii,:],
                                      z[ii,:],Rt,Qt)
              mu.append(new_mu)
              sigma.append(new_sigma)
       
       mu = np.array(mu)
       sigma = np.array(sigma)
       
       x = mu[:,0]
       y = mu[:,1]
       theta = mu[:,2]
       vx = mu[:,3]
       vy = mu[:,4]
       omega = mu[:,5]
       
       # Sending to plot
       figure_1_maker(t,x,y,theta,vx,vy,omega)
       figure_2_maker(x,y,"blue")
       
def Q_e():
       
       global controls , pose
       mu1 = []
       sigma1 = []
       mu2=[]
       sigma2 = []
       
       #Initializing:
       dt = 0.1
       t = np.array(controls[:,0])
       z = np.array(pose[:,1:4])
       u = np.array(controls[:,1:4])
       
       # Initial state vector and cov:
       new_mu1 = np.array([0,0,0,0,0,0])
       new_mu2 = np.array([0,0,0,0,0,0])
       
       new_sigma1 = np.diag([200**2,200**2,200**2,
                             300**2,300**2,400**2])
       new_sigma2 = np.diag([200**2,200**2,200**2,
                             300**2,300**2,400**2])
       
        # Calculating Rt:
       Rt_eps1 = np.array([[10*0.5*dt**2,10*0.5*dt**2,10*0.5*dt**2,
                            10*dt,10*dt,10*dt]]).T
       Rt_eps2 = np.array([[10*0.5*dt**2,10*0.5*dt**2,10*0.5*dt**2,
                            10*dt,10*dt,10*dt]]).T
       # Making Rt:
       
       Rt1 = np.matmul(Rt_eps1,Rt_eps1.T)
       Rt2 = np.matmul(Rt_eps2,Rt_eps2.T)
       
       Qt1 = np.diag([200**2,200**2,200**2])
       Qt2 = np.diag([10**2,10**2,10**2])
       
       # start to build mu (mean) and sigma (cov)
       mu1.append(new_mu1)
       sigma1.append(new_sigma1)
       mu2.append(new_mu2)
       sigma2.append(new_sigma2)

       for ii in range (1,t.shape[0]):
              new_mu1 , new_sigma1 = KF(new_mu1,new_sigma1,u[ii,:],
                                        z[ii,:],Rt1,Qt1)
              new_mu2 , new_sigma2 = KF(new_mu2,new_sigma2,u[ii,:],
                                        z[ii,:],Rt2,Qt2)
              
              mu1.append(new_mu1)
              sigma1.append(new_sigma1)
              mu2.append(new_mu2)
              sigma2.append(new_sigma2)
       
       # Making the cov and the mean into a numpy arrays:
       mu1 = np.array(mu1)
       sigma1 = np.array(sigma1)
       mu2 = np.array(mu2)
       sigma2 = np.array(sigma2)
        
       x1,y1 = [mu1[:,0] , mu1[:,1]]
       x2,y2 = [mu2[:,0] , mu2[:,1]] 

       # Where t = 0,10,20,30,40 sec:
       times = np.array([[0],[100],[200],[300],[400]])
       
       cov1 = []
       cov2 = []
       
       # Getting the cov for t = 0,10,20,30,40 sec.
       for tt in times:
              
              cov = np.array([[sigma1[tt,0,0],sigma1[tt,0,1]],
                              [sigma1[tt,0,1],sigma1[tt,1,1]]])
              cov.resize((2,2))
              cov1.append(cov)
              cov = np.array([[sigma2[tt,0,0],sigma2[tt,0,1]],
                              [sigma2[tt,0,1],sigma2[tt,1,1]]])
              cov.resize((2,2))
              cov2.append(cov)
              
       cov1 = np.array(cov1)
       cov2 = np.array(cov2)
       
       # Sending to plot:
       figure_2_maker(x1,y1,"blue",'e',cov=cov1,mu=mu1[times,[0,1]])
       figure_2_maker(x2,y2,"blue",'e',cov=cov2,mu=mu2[times,[0,1]])

# Getting th data:
global ground_truth , controls , pose
f1 = open("ground_truth.txt",'r')
f2 = open("pose.txt",'r')
f3 = open("controls.txt",'r')

ground_truth = np.array([[np.float64(i) for i in line.split()]\
                          for line in f1])
pose = np.array([[np.float64(i) for i in line.split()]\
                  for line in f2])
controls = np.array([[np.float64(i) for i in line.split()]\
                      for line in f3])

# The menu for question 2 b.) to e.).
menu = 'Please enter b/c/d/e for the respected graphes or q to exit: '
error = "Invalid input. Try again."
       
Ans = input(menu)
while Ans != 'q':
         
       if Ans == 'b':
              Q_b()
       elif Ans == 'c' or Ans == 'd':
              Q_cd(Ans)
       elif Ans =='e':
              Q_e()
       elif Ans == 'q':
              break
       else:
              print (error)
       Ans = input(menu)






