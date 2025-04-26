import matplotlib.pyplot as plt
import numpy as np
class environment:
    def __init__(self,startx=0,starty=0):
        #attempt of abstract simulation
        self.reset(startx=startx,starty=starty)
        self.trajectory=[]
        collisions=True
        while collisions: #ensure that the agent does not spawn on an obstacle
            self.collisions=(np.random.random((30,2))*100).astype(int)
            collisions=self.checkCollisions()
    def reset(self,startx=0,starty=0):
        self.target=[50,50]
        self.map=(100,100)
        self.agent=(startx,starty)
        self.acceleration=0.1
        self.dt=0.01
        self.angle=0
       
    def checkCollisions(self): #check the collisions by looking at euclid didstance
        distances=np.sqrt(np.sum(np.square(self.agent[0]-self.collisions)+np.square(self.agent[1]-self.collisions),axis=1))
        if len(np.where(distances<0.4)[0])>0:
            return True
        return False
    def visualise(self): #visualise the plot - not necesary but good for teaching purposes
        plt.scatter(self.collisions[:,0],self.collisions[:,1],c="b")
        plt.scatter(self.agent[0],self.agent[1],marker="^",c="r")
        plt.scatter(self.target[0],self.target[1],marker="+",c="g")
        traj=np.array(self.trajectory)
        plt.plot(traj[:,0],traj[:,1])
        plt.show()
    def getimage(self,image_width=360,image_height=1,max_blob_size=10,min_dist=0.1): #simulate getting an image
        agent_coords=np.array(self.agent)
        image=np.zeros((image_height,image_width))
        vectors=self.collisions-agent_coords
        distances=np.linalg.norm(vectors,axis=1)
        angles=np.arctan2(vectors[:,1],vectors[:,0])
        angles = (angles+2 *np.pi) % (2*np.pi) % (2*np.pi)
        for dist, angle in zip(distances,angles):
            idx=int((angle/(2*np.pi)) *image_width)
            blob_size=int(max_blob_size/(dist+min_dist))
            blob_size=max(1,blob_size)
            start=max(0,idx-blob_size //2)
            end=min(image_width,idx+blob_size // 2 + 1)
            image[:,start:end]=np.maximum(image[:,start:end],1)
        return image
    def moveAgent(self,velx,vely): #move the agent in the simulator
        #compute linear and angular velocities
        v = (velx + vely) / 2.0  # linear velocity (m/s)
        omega = (velx - vely) #/ wheel_base  # angular velocity (rad/s)
        # update position
        x = v * self.dt * np.cos(self.angle)
        y = v * self.dt * np.sin(self.angle)
        #update orientation
        self.angle += omega * self.dt
        #print(displacementx,displacementy)
        self.agent=[self.agent[0]+x,self.agent[1]+y]
        self.trajectory.append(self.agent)
        return self.checkCollisions()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        self.reset()
        dist=[]
        self.dt=dt
        for t in np.arange(0,T,dt): #loop through timesteps
            vel=agent.step(np.concatenate([self.getimage().flatten(),np.array(self.target)]))  #get agent prediction
            problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent)-np.array(self.target))) #distance to target collection
            if problem: break
        plt.close()
        return np.array(self.trajectory), np.array(dist)
        
if __name__=="__main__": #demo code to show image visualisation over time
    env=environment()
    
    env.dt=0.10
    class agent: #demo agent
        def __init__(self,NUM): #predetermine all the motor positions
            self.velx=(np.abs(np.random.normal(0,1,(NUM,)))+np.random.normal(0,2,(NUM)))
            self.vely=(np.abs(np.random.normal(0,1,(NUM,)))+np.random.normal(0,2,(NUM)))
            self.t=-1
        def step(self,x):
            self.t+=1
            return [self.velx[self.t],self.vely[self.t]]
            
    NUM=1000
    env.runTrial(agent(1000),10)
    """for i in range(NUM):
        problem=env.moveAgent(velx[i],vely[i])
        if problem: break"""
    """plt.cla()
        plt.imshow(env.getimage(image_height=10))
        plt.pause(0.01)"""
    plt.close()
    env.visualise()