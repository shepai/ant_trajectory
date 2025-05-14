import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from GA_code.GAs.GA_MODEL import *
from GA_code.GAs.genetic_algorithm import *
from grid_environment import environment
import cv2

env=environment() #call in demo environment
#setup the network parameters so that the input is all the pixels values and output 2 velocities for motors
_INPUT_SIZE_=cv2.resize(env.getObservation(), (8, 48), interpolation = cv2.INTER_AREA).shape[:2] #the 2 is for the target location, but realistically it will not know where the target is
_OUTPUTSIZE_=3

def fitness(trajectory, distances):
    # Reward reducing distance to the target
    improvement = distances[0] - distances[-1]
    improvement = max(improvement, 0)  # clip negative improvements
    return improvement

#microbial GA
ga=Microbial_GA(50,50,0.2,sex=0) #
ga.initialize_population(controllerCNN_LRF,[_INPUT_SIZE_,200,_OUTPUTSIZE_],std=0.3)
print("Begin trial")
history,fitness=ga.evolve(env,fitness,outputs=True) #run the GA
print(fitness.shape)
#from here you will want to find the best one from ga population from our fitness matrix
best_genotype=ga.pop[np.argmax(fitness)]

del env
env=environment(record=1)
env.runTrial(best_genotype)
env.out.release()
env.visualise()
