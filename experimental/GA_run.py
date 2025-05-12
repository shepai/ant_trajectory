import sys
sys.path.insert(1,"/its/home/drs25/Documents/GitHub/ant_trajectory") #put path here
from GA_code.GAs.GA_MODEL import *
from GA_code.GAs.genetic_algorithm import *
from grid_environment import environment


env=environment() #call in demo environment

#setup the network parameters so that the input is all the pixels values and output 2 velocities for motors
_INPUT_SIZE_=env.getObservation().flatten().shape[0]+2 #the 2 is for the target location, but realistically it will not know where the target is
_H1_=16
_H2_=6
_OUTPUTSIZE_=3

def fitness(trajectory,targets):
    #for this example our goal is to be getting closer to the target and ending up there
    #so we can use the fitness as a sum of all values as it increases
    correct=np.sum(np.diff(targets)*-1)
    fitness=correct/len(np.arange(0,1,0.1)) #total moves getting closer / total time
    if fitness<0: fitness=0
    return fitness

#microbial GA
ga=Microbial_GA(100,10,0.2,sex=1) #
ga.initialize_population(controller_LRF,[_INPUT_SIZE_,[_H1_,_H2_],_OUTPUTSIZE_])
print("Begin trial")
history,fitness=ga.evolve(env,fitness,outputs=True) #run the GA
print(fitness.shape)
#from here you will want to find the best one from ga population from our fitness matrix
best_genotype=ga.pop[np.argmax(fitness)]
env.runTrial(best_genotype)
env.visualise()
