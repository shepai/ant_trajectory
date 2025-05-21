import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import time
try:
    from autoencoder.encode import encode
except:
    print("error starting auto encoder")
#matplotlib.use('TkAgg')
class environment:
    def __init__(self,data="/data/full_arena_grid_infer_views/",show=0,record=0,filename="output.avi",randomize_start=False):
        #form the correct datapaths
        self.filename=filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path=script_dir.replace("grid_environment","")
        self.datapath=self.path+data
        self.dt=0.1
        self.grid=pd.read_csv(self.datapath+"/full_grid_views_meta_data.csv")
        self.x=self.grid["x_m"]
        self.y=self.grid["y_m"]
        self.files={}
        for f in self.grid['img_name']:
            self.files[f]=cv2.resize(cv2.cvtColor(cv2.imread(self.datapath+'/'+f).astype(np.uint8), cv2.COLOR_BGR2GRAY), (8, 48), interpolation = cv2.INTER_AREA).T
        self.show=show
        self.record=record
        self.recording=0
        self.target=(0.15,-0.003) #food source
        self.random=randomize_start
        self.reset()
    def reset(self):
        self.agent_pos=[0.08,0.6]
        if self.random: self.agent_pos=[np.random.uniform(np.min(self.x), np.max(self.x)),np.random.uniform(np.min(self.y), np.max(self.y))]
        self.angle=0
        self.trajectory=[]
        self.prev_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target)) # this saves the initial distance from the agent to the target for comparison after each step
        if self.recording:
            self.out.release()
            self.recording=0
        if self.record:
            frame = self.getObservation()
            if frame is None:
                raise RuntimeError("Failed to get initial observation for video recording.")
            if frame.dtype != np.uint8:
                raise ValueError(f"Expected shape (H, W) uint8 but got {frame.shape}, {frame.dtype}")
            
            self.recording = 1
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(
                self.filename,
                fourcc, 20.0, (width, height)
            )
            if not self.out.isOpened():
                raise RuntimeError("VideoWriter failed to open. Check codec, path, and frame size.")
            
    def getObservation(self):
        image=self.find_nearest(*self.agent_pos)
        pixels_per_degree = image.shape[1] / 360.0
        offset = int(pixels_per_degree * np.degrees(self.angle))
        # Shift image horizontally (wrap around using numpy.roll)
        rotated = np.roll(image, -offset, axis=1)  # negative for clockwise
        #rotated=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        return rotated
    def find_nearest(self,x,y):
        closest_x = min(range(len(self.x)), key=lambda i: abs(self.x[i] - x))
        closest_y = min(range(len(self.y)), key=lambda i: abs(self.y[i] - y))
        x_val = self.x[closest_x]
        y_val = self.y[closest_y]
        condition = np.isclose(self.grid['x_m'], x_val) & np.isclose(self.grid['y_m'], y_val)
        nearest_row = self.grid[condition]
        if not nearest_row.empty:
            return self.files[nearest_row['img_name'].values[0]]
        else:
            print("Error: Cooked *skull face emoji*")
            return None
    def moveAgent(self,x,y):
        v = (x + y) / 2.0  # linear velocity (m/s)
        omega = (x - y) #/ wheel_base  # angular velocity (rad/s)
        # update position
        x = max(x,v) * self.dt * np.cos(self.angle)
        y = max(y,v) * self.dt * np.sin(self.angle)
        #update orientation
        self.angle += omega * self.dt
        if self.angle>360: #wrap round
            self.angle=self.angle-360
        self.agent_pos[0]+=x
        self.agent_pos[1]+=y
        self.trajectory.append(self.agent_pos.copy())
        image=self.getObservation()
        if self.show:
            plt.cla()
            plt.imshow(image)
            plt.pause(0.01)
        if self.record and self.recording:
            if image is not None and image.dtype == np.uint8:
                #print("Writing frame:", image.shape, image.dtype)
                bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.out.write(bgr_frame)
            else:
                print("Skipping frame — invalid shape or type:", image)
        if np.average(image)<25 or self.died(): #too dark
            return True
        return False
    def died(self):
        x,y=self.agent_pos
        if x<np.min(self.x) or y<np.min(self.y) or x>np.max(self.x) or y>np.max(self.y):
            #print(x,y,np.min(self.x),np.min(self.y),np.max(self.x),np.max(self.y))
            return True
        return False
    def visualise(self):
        plt.close()
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        t_=time.time()
        self.reset()
        self.trajectory.append(self.agent_pos.copy())
        dist=[]
        self.dt=dt
        for t in np.arange(0,T,dt): #loop through timesteps
            observation = self.getAntVision()
            observation=observation.reshape((1,*observation.shape,1)) if "CNN" in str(agent.__class__) else encode(observation)
            vel=agent.step(observation/255)  #get agent prediction #ODO update for CNN
            if "LRF" in str(agent.__class__):
                options=[[0,1.5],[1.5,0],[.1,.1]]
                problem=self.moveAgent(*options[vel]) #move to target
            else: 
                problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent_pos)-np.array(self.target))) #distance to target collection
            if problem: break
        #print("\tRan trial in",(time.time()-t_),"seconds")
        return np.array(self.trajectory), np.array(dist)
    def getAntVision(self):
        observation=self.getObservation()#observation = cv2.resize(self.getObservation(), (8, 48), interpolation = cv2.INTER_AREA)
        return observation.reshape((1,*observation.shape))
        
    def step(self,action):
        # Action map: 0=right, 1=left, 2=forward
        options=[[0,0.5],[0.5,0],[0.1,0.1]]
        vel=options[action]

        prev_pos = np.array(self.agent_pos)
        prev_dist = np.linalg.norm(prev_pos - np.array(self.target))
        
        done=self.moveAgent(*vel) #move agent
        observation = self.getAntVision()
        traj=np.array(self.trajectory)
        #@alej this is how I have put in reward but feel free to change it
        #reward=np.linalg.norm(traj[0]-traj[-1]) # @dex I seem to understand that there is a larger reward for covering more distance rather than getting closer to the food

        #trying vector to goal 
        vec_to_goal = np.array(self.target) - np.array(self.agent_pos)
        unit_vec = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-6)

        curr_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target))
        distance_reward = prev_dist - curr_distance
    
        # Directional reward
        vec_to_goal = np.array(self.target) - np.array(self.agent_pos)
        unit_vec = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-6)
        heading_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        directional_reward = np.dot(unit_vec, heading_vec)
    
        # Combine rewards
        reward = 10 * distance_reward + 2 * directional_reward

        
        # curr_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target)) #I hope this works... It should basically calculate a reward based on how closer to the target the agent gets
        # reward = self.prev_distance - curr_distance  # positive if getting closer
        self.prev_distance = curr_distance
        # Optional penalty for dying 
        if self.died():
            reward -= 10  # strong negative penalty. Don't die lil ant
        info={}
        return observation,reward,done,info


import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,data="/data/full_arena_grid_infer_views/",show=0,record=0,filename="output.avi",randomize_start=False):
        super().__init__()
        #form the correct datapaths
        self.filename=filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path=script_dir.replace("grid_environment","")
        self.datapath=self.path+data
        self.dt=0.1
        self.grid=pd.read_csv(self.datapath+"/full_grid_views_meta_data.csv")
        self.x=self.grid["x_m"]
        self.y=self.grid["y_m"]
        self.files={}
        for f in self.grid['img_name']:
            self.files[f]=cv2.resize(cv2.cvtColor(cv2.imread(self.datapath+'/'+f).astype(np.uint8), cv2.COLOR_BGR2GRAY), (8, 48), interpolation = cv2.INTER_AREA).T
        self.show=show
        self.record=record
        self.recording=0
        self.target=(0.15,-0.003) #food source
        self.random=randomize_start
        self.reset()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation space: 48x8 grayscale image (1 channel), dtype uint8
        self.observation_space = spaces.Box(low=0, high=255, shape=(1 ,8, 48), dtype=np.uint8)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos=[0.08,0.6]
        if self.random: self.agent_pos=[np.random.uniform(np.min(self.x), np.max(self.x)),np.random.uniform(np.min(self.y), np.max(self.y))]
        self.angle=0
        self.trajectory=[]
        self.prev_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target)) # this saves the initial distance from the agent to the target for comparison after each step
        if self.recording:
            self.out.release()
            self.recording=0
        if self.record:
            frame = self.getObservation()
            if frame is None:
                raise RuntimeError("Failed to get initial observation for video recording.")
            if frame.dtype != np.uint8:
                raise ValueError(f"Expected shape (H, W) uint8 but got {frame.shape}, {frame.dtype}")
            
            self.recording = 1
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(
                self.filename,
                fourcc, 20.0, (width, height)
            )
            if not self.out.isOpened():
                raise RuntimeError("VideoWriter failed to open. Check codec, path, and frame size.")
        return self.getObservation().reshape((1 ,8, 48)).astype(np.uint8),{}
    def getObservation(self):
        image=self.find_nearest(*self.agent_pos)
        pixels_per_degree = image.shape[1] / 360.0
        offset = int(pixels_per_degree * np.degrees(self.angle))
        # Shift image horizontally (wrap around using numpy.roll)
        rotated = np.roll(image, -offset, axis=1)  # negative for clockwise
        #rotated=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        return rotated
    def find_nearest(self,x,y):
        closest_x = min(range(len(self.x)), key=lambda i: abs(self.x[i] - x))
        closest_y = min(range(len(self.y)), key=lambda i: abs(self.y[i] - y))
        x_val = self.x[closest_x]
        y_val = self.y[closest_y]
        condition = np.isclose(self.grid['x_m'], x_val) & np.isclose(self.grid['y_m'], y_val)
        nearest_row = self.grid[condition]
        if not nearest_row.empty:
            return self.files[nearest_row['img_name'].values[0]]
        else:
            print("Error: Cooked *skull face emoji*")
            return None
    def moveAgent(self,x,y):
        v = (x + y) / 2.0  # linear velocity (m/s)
        omega = (x - y) #/ wheel_base  # angular velocity (rad/s)
        # update position
        x = max(x,v) * self.dt * np.cos(self.angle)
        y = max(y,v) * self.dt * np.sin(self.angle)
        #update orientation
        self.angle += omega * self.dt
        if self.angle>360: #wrap round
            self.angle=self.angle-360
        self.agent_pos[0]+=x
        self.agent_pos[1]+=y
        self.trajectory.append(self.agent_pos.copy())
        image=self.getObservation()
        if self.show:
            plt.cla()
            plt.imshow(image)
            plt.pause(0.01)
        if self.record and self.recording:
            if image is not None and image.dtype == np.uint8:
                #print("Writing frame:", image.shape, image.dtype)
                bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.out.write(bgr_frame)
            else:
                print("Skipping frame — invalid shape or type:", image)
        if np.average(image)<25 or self.died(): #too dark
            return True
        return False
    def died(self):
        x,y=self.agent_pos
        if x<np.min(self.x) or y<np.min(self.y) or x>np.max(self.x) or y>np.max(self.y):
            #print(x,y,np.min(self.x),np.min(self.y),np.max(self.x),np.max(self.y))
            return True
        return False
    def visualise(self):
        plt.close()
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def getAntVision(self):
        observation=self.getObservation()#observation = cv2.resize(self.getObservation(), (8, 48), interpolation = cv2.INTER_AREA)
        return observation.reshape((1 ,8, 48)).astype(np.uint8)
        
    def step(self,action):
        # Action map: 0=right, 1=left, 2=forward
        prev_pos = np.array(self.agent_pos)
        prev_dist = np.linalg.norm(prev_pos - np.array(self.target))
        done=self.moveAgent(*action) #move agent
        observation = self.getAntVision()
        traj=np.array(self.trajectory)
        #@alej this is how I have put in reward but feel free to change it
        #reward=np.linalg.norm(traj[0]-traj[-1]) # @dex I seem to understand that there is a larger reward for covering more distance rather than getting closer to the food

        #trying vector to goal 
        vec_to_goal = np.array(self.target) - np.array(self.agent_pos)
        unit_vec = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-6)

        curr_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target))
        distance_reward = prev_dist - curr_distance
    
        # Directional reward
        vec_to_goal = np.array(self.target) - np.array(self.agent_pos)
        unit_vec = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-6)
        heading_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        directional_reward = np.dot(unit_vec, heading_vec)
    
        # Combine rewards
        reward = 10 * distance_reward + 2 * directional_reward

        
        # curr_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target)) #I hope this works... It should basically calculate a reward based on how closer to the target the agent gets
        # reward = self.prev_distance - curr_distance  # positive if getting closer
        self.prev_distance = curr_distance
        # Optional penalty for dying 
        if self.died():
            reward -= 10  # strong negative penalty. Don't die lil ant
        info={}
        return observation.reshape((1 ,8, 48)).astype(np.uint8),reward,done,False,info

    def render(self):
        # Optional visualization
        pass

    def close(self):
        # Cleanup if needed
        pass
if __name__=="__main__":
    env=CustomEnv(show=1,record=0)
    #env=environment(show=1,record=1)
    """import keyboard
    import time
    x, y = 0.0, 0.0
    step = 0.1
    env.reset()
    try:
        while True:
            if keyboard.is_pressed('up'):
                env.moveAgent(step, 0)
            elif keyboard.is_pressed('down'):
                env.moveAgent(-step, 0)
            if keyboard.is_pressed('right'):
                env.moveAgent(0, step)
            elif keyboard.is_pressed('left'):
                env.moveAgent(0, -step)
            #env.moveAgent(0, 0)
            time.sleep(0.01)  # small delay to avoid flooding with commands
    except KeyboardInterrupt:
        env.out.release()
        plt.close()
        env.visualise()"""
    """ar=[]
    c=0
    for file in env.files:
        for theta in range(360):
            print(c,"images processed")
            image=env.files[file].copy()
            pixels_per_degree = image.shape[1] / 360.0
            offset = int(pixels_per_degree * np.degrees(theta))
            # Shift image horizontally (wrap around using numpy.roll)
            rotated = np.roll(image, -offset, axis=1)  # negative for clockwise
            ar.append(image)
            c+=1

    ar=np.array(ar)
    print(ar.shape)"""
    #np.save("/its/home/drs25/ant_trajectory/autoencoder/allangles",ar)

        
    
