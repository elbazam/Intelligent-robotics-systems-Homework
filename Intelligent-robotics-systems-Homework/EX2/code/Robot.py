import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from Ploter import config_plot
from World import World



class Robot:
    """
    the robot class, we will use this to describe a robot
    """
    def __init__(self, world_size=100):
        """
        creating a robot object
        :param world_size: the world size in pixels
        """
        self._world_size = world_size
        # pose declaration
        self.x = np.random.rand() * self._world_size
        self.y = np.random.rand() * self._world_size
        self.theta = np.random.rand() * 2 * np.pi
        # noise declaration
        self.forward_noise = 0
        self.turn_noise = 0
        self.sense_distance_noise = 0

    def set(self, new_x, new_y, new_orientation):
        """
        setting the configuration of the robot
        :param new_x: the new x coordinate
        :param new_y: the new y coordinate
        :param new_orientation: the new orientation
        """
        if new_x < 0 or new_x >= self._world_size:
            raise Exception('X coordinate out of bound')

        if new_y < 0 or new_y >= self._world_size:
            raise Exception('Y coordinate out of bound')

        if new_orientation < 0.0 or new_orientation >= 2 * np.pi:
            Exception('Orientation must be in [0,2pi]')

        self.x = new_x
        self.y = new_y
        self.theta = new_orientation

    def print(self):
        """"
        printing the pose
        """
        print('[x= {} y={} heading={}]'.format(self.x, self.y, self.theta))

    def plot(self, mycolor="b", style="robot", show=True, markersize=1):
        """
        plotting the pose of the robot in the world
        :param mycolor: the color of the robot
        :param style: the style to plot with
        :param show: if to show or not show - used to create a new figure or not
        """
        if style == "robot":
            phi = np.linspace(0, 2 * np.pi, 101)
            r = 1
            # plot robot body
            plt.plot(self.x + r * np.cos(phi), self.y + r * np.sin(phi), color=mycolor)
            # plot heading direction
            plt.plot([self.x, self.x + r * np.cos(self.theta)], [self.y, self.y + r * np.sin(self.theta)], color=mycolor)

        elif style == "particle":
            plt.plot(self.x, self.y, '.', color=mycolor, markersize=markersize)
        else:
            print("unknown style")

        if show:
            plt.show()

    def set_noise(self, new_forward_noise, new_turn_noise,new_sense_noise_range, new_sense_noise_bearing):
        """
        setting the noise if pose of the robot
        :param new_forward_noise: the noise for moving forward
        :param new_turn_noise: the noise in the turn of the robot
        :param new_sense_noise_range: the noise in range measurement
        :param new_sense_noise_bearing: the noise in bearing measurement
        """
        self.forward_noise = new_forward_noise
        self.turn_noise = new_turn_noise
        self.sense_noise_range = new_sense_noise_range
        self.sense_noise_bearing = new_sense_noise_bearing


    def get_pose(self):
        """
        returning the pose vector
        :return: (x, y, theta) the pose vector
        """
        return self.x, self.y, self.theta
    
    ''' ================ Added code ================= '''
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
        
        
        


