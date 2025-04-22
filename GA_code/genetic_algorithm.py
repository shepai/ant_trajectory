from GA_MODEL import controller
import numpy as np
from copy import deepcopy
def fitness(positions,target): #fitness that takes in positions over time and the target
    #return the fitness based on how quickly it got to the target
    return 0

_INPUT_SIZE_=10
_H1_=10
_H2_=10
_OUTPUTSIZE_=10

class Microbial_GA:
    def __init__(self,num_generations,population_size,mutation_rate,sex=False):
        """
        Genetic algorithm class for running the GA
        It will need some changes to better match the environment you make
        """
        self.generations=num_generations
        self.pop_zise=population_size
        self.rate=mutation_rate
        self.initialize_population()
        self.sex=sex
    def initialize_population(self):
        population=[]
        for i in range(self.pop_zise):
            population.append(controller(_INPUT_SIZE_,[_H1_,_H2_],_OUTPUTSIZE_))   #@seyi this is where the network sizes go
        self.pop=population
    def evolve(self,environment):
        history=[]
        fitness_matrix=np.zeros((self.pop_zise))
        for i in range(self.pop_zise): #calculate current fitness of all genotypes
            positions,target=environment.runTrial(self.pop[i])
            fitness_matrix[i]=fitness(positions,target)

        for gen in range(self.generations): #begin actual evolution
            ind1=np.random.randint(0,self.pop_zise-1)
            ind2=ind1
            while ind2==ind1: #make sure second geno is not first
                ind2=np.random.randint(0,self.pop_zise-1)
            if fitness_matrix[ind2]>fitness_matrix[ind1]: #selection
                self.pop[ind1]=deepcopy(self.pop[ind2])
                if self.sex: self.pop[ind1].sex(self.pop[ind1],self.pop[ind2])
                self.pop[ind1].mutate() #mutate
                positions,target=environment.runTrial(self.pop[ind1])
                fitness_matrix[ind1]=fitness(positions,target)
            elif fitness_matrix[ind1]>fitness_matrix[ind2]: #selection
                self.pop[ind2]=deepcopy(self.pop[ind1])
                self.pop[ind2].mutate() #mutate
                if self.sex: self.pop[ind2].sex(self.pop[ind2],self.pop[ind1])
                positions,target=environment.runTrial(self.pop[ind2])
                fitness_matrix[ind2]=fitness(positions,target)
            history.append(np.max(fitness_matrix))
        return history,fitness_matrix
    
if __name__ == "__main__": #test code to run in same file 
    def fitness(positions,target):
        return np.random.random((1))
    class environment_example:
        def __init__(self):
            pass
        def runTrial(self,geno):
            return [0,0,0,0],[0,0,0,0]
    ga=Microbial_GA(100,10,0.2,sex=1)
    history,fitness=ga.evolve(environment_example())
    print(history)