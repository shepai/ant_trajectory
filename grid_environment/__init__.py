import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
class environment:
    def __init__(self,data="/data/full_arena_grid_infer_views/",show=0,record=0):
        #form the correct datapaths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path=script_dir.replace("grid_environment","")
        self.datapath=self.path+data
        self.dt=0.1
        self.grid=pd.read_csv(self.datapath+"/full_grid_views_meta_data.csv")
        print(self.grid.head())
        self.x=self.grid["x_m"]
        self.y=self.grid["y_m"]
        self.show=show
        self.record=record
        self.recording=0
        self.reset()
    def reset(self):
        self.agent_pos=[-0.170,-0.443]
        self.angle=360
        self.trajectory=[]
        if self.recording:
            self.out.release()
            self.recording=0
        if self.record:
            self.recording=1
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame=self.getObservation()
            print(frame.shape)
            self.out = cv2.VideoWriter(self.path+'/data/video_generator/output.avi', fourcc, 20.0, frame.shape)  # 20 FPS, 640x480 resolution

    def getObservation(self):
        image=self.find_nearest(*self.agent_pos)
        pixels_per_degree = image.shape[1] / 360.0
        offset = int(pixels_per_degree * self.angle)
        # Shift image horizontally (wrap around using numpy.roll)
        rotated = np.roll(image, -offset, axis=1)  # negative for clockwise
        return rotated
    def find_nearest(self,x,y):
        closest_x = min(range(len(self.x)), key=lambda i: abs(self.x[i] - x))
        closest_y = min(range(len(self.y)), key=lambda i: abs(self.y[i] - y))
        x_val = self.x[closest_x]
        y_val = self.y[closest_y]
        condition = np.isclose(self.grid['x_m'], x_val) & np.isclose(self.grid['y_m'], y_val)
        nearest_row = self.grid[condition]
        if not nearest_row.empty:
            return cv2.imread(self.datapath+'/'+nearest_row['img_name'].values[0])
        else:
            return None
    def moveAgent(self,x,y):
        v = (x + y) / 2.0  # linear velocity (m/s)
        omega = (x - y) #/ wheel_base  # angular velocity (rad/s)
        # update position
        x = v * self.dt * np.cos(self.angle)
        y = v * self.dt * np.sin(self.angle)
        #update orientation
        self.angle += omega * self.dt
        self.agent_pos[0]+=y
        self.agent_pos[1]+=x
        self.trajectory.append(self.agent_pos.copy())
        if self.show:
            plt.cla()
            plt.imshow(self.getObservation())
            plt.pause(0.01)
        if self.record:
            self.out.write(self.getObservation())
    def visualise(self):
        plt.close()
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        self.reset()
        dist=[]
        self.dt=dt
        for t in np.arange(0,T,dt): #loop through timesteps
            vel=agent.step(np.concatenate([self.getimage().flatten(),np.array(self.target)]))  #get agent prediction
            if "LRF" in str(agent.__class__):
                options=[[0,0.01],[0.01,0],[0.01,0.01]]
                problem=self.moveAgent(*options[vel]) #move to target
            else: 
                problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent)-np.array(self.target))) #distance to target collection
            if problem: break
        return np.array(self.trajectory), np.array(dist)
        
if __name__=="__main__":
    env=environment(show=1,record=0)
    import keyboard
    import time
    x, y = 0.0, 0.0
    step = 0.01
    try:
        while True:
            if keyboard.is_pressed('up'):
                y += step
            elif keyboard.is_pressed('down'):
                y -= step
            elif keyboard.is_pressed('right'):
                x += step
            elif keyboard.is_pressed('left'):
                x -= step

            env.moveAgent(x, y)
            time.sleep(0.01)  # small delay to avoid flooding with commands
    except KeyboardInterrupt:
        plt.close()
        env.visualise()
    