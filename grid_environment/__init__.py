import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')
class environment:
    def __init__(self,data="/data/full_arena_grid_infer_views/",show=0,record=0):
        #form the correct datapaths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path=script_dir.replace("grid_environment","")
        self.datapath=self.path+data
        self.dt=0.1
        self.grid=pd.read_csv(self.datapath+"/full_grid_views_meta_data.csv")
        self.x=self.grid["x_m"]
        self.y=self.grid["y_m"]
        self.show=show
        self.record=record
        self.recording=0
        self.reset()
        self.target=(0.15,-0.003) #food source
    def reset(self):
        self.agent_pos=[-0.170,-0.443]
        self.angle=0
        self.trajectory=[]
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
                self.path + '/data/video_generator/output2.avi',
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
        rotated=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
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
            print("Error: Cooked *skull face emoji*")
            return None
    def moveAgent(self,x,y):
        v = (x + y) / 2.0  # linear velocity (m/s)
        omega = (x - y) #/ wheel_base  # angular velocity (rad/s)
        # update position
        x = v * self.dt * np.cos(self.angle)
        y = v * self.dt * np.sin(self.angle)
        #update orientation
        self.angle += omega * self.dt
        if self.angle>360: #wrap round
            self.angle=self.angle-360
        self.agent_pos[0]+=y
        self.agent_pos[1]+=x
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
    def visualise(self):
        plt.close()
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        t_=time.time()
        self.reset()
        dist=[]
        self.dt=dt
        for t in np.arange(0,T,dt): #loop through timesteps
            observation= self.getObservation().reshape((1,*self.getObservation().shape,1)) if "CNN" in str(agent.__class__) else np.concatenate([self.getObservation().flatten(),np.array(self.target)])
            vel=agent.step(observation)  #get agent prediction #ODO update for CNN
            if "LRF" in str(agent.__class__):
                options=[[0,0.01],[0.01,0],[0.01,0.01]]
                problem=self.moveAgent(*options[vel]) #move to target
            else: 
                problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent_pos)-np.array(self.target))) #distance to target collection
            if problem: break
        print("\tRan trial in",(time.time()-t_),"seconds")
        return np.array(self.trajectory), np.array(dist)
        
if __name__=="__main__":
    env=environment(show=1,record=1)
    import keyboard
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
        env.visualise()
        
    