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
        self.acceleration=0.2
        
       
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
        self.agent=[self.agent[0]+(self.acceleration*velx),self.agent[1]+(self.acceleration*vely)]
        self.trajectory.append(self.agent)
        return self.checkCollisions()
    def runTrial(self,agent,T=1,dt=0.01): #run a trial
        self.reset()
        dist=[]
        for t in np.arange(0,T,dt): #loop through timesteps
            vel=agent.step(np.concatenate([self.getimage().flatten(),np.array(self.target)]))  #get agent prediction
            vel=vel/10
            vel[vel<-5]=-5
            vel[vel>5]=5
            problem=self.moveAgent(vel[0],vel[1]) #move to target
            dist.append(np.linalg.norm(np.array(self.agent)-np.array(self.target))) #distance to target collection
            if problem: break
        return np.array(self.trajectory), np.array(dist)
        
if __name__=="__main__": #demo code to show image visualisation over time
    env=environment()
    velx=sorted(np.random.normal(0,5,(200,)))
    vely=sorted(np.random.normal(0,5,(120,)))
    print(velx)
    for i in range(100):
        problem=env.moveAgent(velx[i],vely[i])
        if problem: break
        plt.cla()
        plt.imshow(env.getimage())
        plt.pause(0.05)
    plt.close()
    env.visualise()