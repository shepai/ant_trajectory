import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

class environment:
    def __init__(self,startx=0,starty=0,sho=False):
        #start pybullet
        if sho: p.connect(p.GUI,options="--disable-rendering --silent")
        else: p.connect(p.DIRECT,options="--disable-rendering --silent")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #load environment
        self.plane=p.loadURDF("plane.urdf") #@Seyij this is for the urdf of the ant environment
        p.setGravity(0, 0, -9.8)
        self.robot_id = p.loadURDF("husky/husky.urdf", [startx, starty, 0.1]) #feel free to replace this robot if you have a better one
        self.startx=startx
        self.starty=starty
        self.reset()
        self.target=[50,50] #you can set this target to vvhatever you vvant. 
        self.agent=[startx,starty]
        self.sho=sho
    def reset(self):
        p.resetSimulation()
        self.trajectories=[]
        self.plane=p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)
        self.robot_id = p.loadURDF("husky/husky.urdf", [self.startx, self.starty, 0.1])
        for i in range(100):
            p.setTimeStep(1. / 240.)  # Typical time step for smooth simulation
            p.stepSimulation()
    def getimage(self):
        # Get robot position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = np.degrees(euler[2])  # Extract yaw angle from the robot's orientation

        # Camera target set relative to the robot's current position and yaw
        cam_target = [pos[0] + np.cos(np.radians(yaw)), pos[1] + np.sin(np.radians(yaw)), pos[2]]
        cam_up = [0, 0, 1]  # Up direction for the camera (Z-axis)

        # View matrix for the camera (camera slightly above the robot)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[pos[0], pos[1], pos[2] + 0.5],  # Position above the robot
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up
        )

        # Projection matrix with a wide FOV (120 degrees for panoramic effect)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=120,               # Wide Field of View for panoramic effect
            aspect=1.0,            # Aspect ratio of the image
            nearVal=0.01,          # Near clipping plane
            farVal=10              # Far clipping plane
        )

        # Capture the image from the camera
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=320,          # Image width
            height=320,         # Image height
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        return rgb_img
    def moveAgent(self,velx,vely):
        wheel_joints = [2, 3, 4, 5]  # Example wheel joints
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=wheel_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[velx,vely,velx,vely]  # Set speed to 5 for all wheels
        )
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.trajectories.append([pos[0],pos[1]])
        self.agent=[pos[0],pos[1]]
        p.setTimeStep(1. / 24.)  # Typical time step for smooth simulation
        p.stepSimulation()
    def visualise(self): #visualise the plot - not necesary but good for teaching purposes
        plt.scatter(self.collisions[:,0],self.collisions[:,1],c="b")
        plt.scatter(self.agent[0],self.agent[1],marker="^",c="r")
        plt.scatter(self.target[0],self.target[1],marker="+",c="g")
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        self.reset()
        dist=[]
        self.dt=dt
        for t in np.arange(0,T,dt): #loop through timesteps
            vel=agent.step(np.concatenate([self.getimage().flatten(),np.array(self.target)]))  #get agent prediction
            problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent)-np.array(self.target))) #distance to target collection
            if problem: break
            if self.sho:
                cv2.imshow("Panoramic Camera View", self.getimage())
                cv2.waitKey(1) 
        return np.array(self.trajectory), np.array(dist)

    def close(self):
        p.disconnect()

if __name__=="__main__": #demo code to show image visualisation over time
    env=environment(sho=True)
    
    env.dt=0.10
    class agent: #demo agent
        def __init__(self,NUM): #predetermine all the motor positions
            self.velx=np.ones((NUM,))*5#(np.abs(np.random.normal(0,1,(NUM,)))+np.random.normal(0,2,(NUM)))
            self.vely=np.ones((NUM,))*-5#(np.abs(np.random.normal(0,1,(NUM,)))+np.random.normal(0,2,(NUM)))
            self.t=-1
        def step(self,x):
            self.t+=1
            return [self.velx[self.t],self.vely[self.t]]
            
    NUM=1000
    
    env.runTrial(agent(1000),10)
    time.sleep(5)
    env.close()