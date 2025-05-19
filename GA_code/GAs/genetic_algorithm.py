
import numpy as np
from copy import deepcopy
import random

class GA:
    def __init__(self,num_generations,population_size,mutation_rate,sex=False):
        """
        Genetic algorithm class for running the GA
        It will need some changes to better match the environment you make
        """
        self.generations=num_generations
        self.pop_zise=population_size
        self.rate=mutation_rate
        #self.initialize_population(controller,[_INPUT_SIZE_,[_H1_,_H2_],_OUTPUTSIZE_])
        self.sex=sex
    def initialize_population(self,contr,params,std=0.2):
        self.contr=contr
        self.params=params
        population=[]
        for i in range(self.pop_zise):
            population.append(contr(*params,std=std))   #@seyi this is where the network sizes go
        self.pop=population
class Microbial_GA(GA):
    def evolve(self,environment,fitness,outputs=False):
        history=[]
        fitness_matrix=np.zeros((self.pop_zise))
        for i in range(self.pop_zise): #calculate current fitness of all genotypes
            positions,target=environment.runTrial(self.pop[i])
            fitness_matrix[i]=fitness(positions,target)

        for gen in range(self.generations): #begin actual evolution
            if outputs:
                print("Generation",gen,"best fitness:",np.max(fitness_matrix))
            ind1=np.random.randint(0,self.pop_zise-1)
            ind2=ind1
            while ind2==ind1: #make sure second geno is not first
                ind2=np.random.randint(0,self.pop_zise-1)
            if fitness_matrix[ind2]>=fitness_matrix[ind1]: #selection
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
    
class NEAT(GA):
    def compatibility_distance(self,genome1, genome2, c=1.0):
        """
        Calculates a similarity score between two genomes using Euclidean distance.
        """
        genome1 = np.array(genome1)
        genome2 = np.array(genome2)
        if len(genome1) != len(genome2): #different species
            dist=np.inf
        dist = np.linalg.norm(genome1 - genome2)
        return c * dist
    def assign_species(self, threshold=3.0):
        species = []
        representatives = []

        for i, genome in enumerate(self.pop):
            placed = False
            for j, rep in enumerate(representatives):
                if self.compatibility_distance(genome.geno, rep.geno) < threshold:
                    species[j].append(i)
                    placed = True
                    break
            if not placed:
                representatives.append(genome)
                species.append([i])
        
        return species
    def compute_adjusted_fitness(self,population, fitnesses, species_list):
        adjusted = np.zeros(len(population))
        for species in species_list:
            size = len(species)
            for i in species:
                adjusted[i] = fitnesses[i] / size
        return adjusted
    def select_parent(self,species, adjusted_fitnesses):
        fitness_vals = [adjusted_fitnesses[i] for i in species]
        total = sum(fitness_vals)
        if total == 0:
            # All fitnesses are zero — pick randomly
            return random.choice(species)

        probs = [f / total for f in fitness_vals]
        return random.choices(species, weights=probs, k=1)[0]
    def evolve(self,environment,fitness,outputs=False,rate=0.2):
        history=[0]
        for gen in range(self.generations):
            if outputs:
                print("Generation",gen,"best fitness:",np.max(history))
            fitness_matrix=np.zeros((self.pop_zise))
            for i in range(self.pop_zise): #calculate current fitness of all genotypes
                positions,target=environment.runTrial(self.pop[i])
                fitness_matrix[i]=fitness(positions,target)
            history.append(np.max(fitness_matrix))
            species_list=self.assign_species()
            adj=self.compute_adjusted_fitness(self.pop,fitness_matrix,species_list)
            new_population = []
            elitism=1
            for species in species_list:
                # Sort individuals by fitness
                sorted_species = sorted(species, key=lambda i: fitness_matrix[i], reverse=True)

                # --- ELITISM ---
                elites = [deepcopy(self.pop[i]) for i in sorted_species[:elitism]]
                new_population.extend(elites)
                prob_winning=0.5
                # --- BREEDING ---
                num_offspring = len(species) - elitism
                for _ in range(num_offspring):
                    parent1 = self.select_parent(species, adj)
                    if np.random.rand() < prob_winning:
                        parent2 = self.select_parent(species, adj)
                        offspring = self.pop[parent1].sex(self.pop[parent1],self.pop[parent2]) 
                    else:
                        offspring = deepcopy(self.pop[parent1])

                    # --- MUTATION ---
                    offspring.mutate(rate=rate)
                    if np.random.rand() < rate: #rmove a layer
                        offspring.delete_layer()
                    if np.random.rand() < rate: #add a layer
                        offspring.insert_layer(np.random.normal(0,offspring.std,(512,512)),np.random.normal(0,offspring.std,(512,)))
                    
                    new_population.append(offspring)
        return np.array(history),fitness_matrix

class Differential(GA):
    def evolve(self,environment,fitness,outputs=False,rate=0.2):
        history=[0]
        fitness_matrix=np.zeros((self.pop_zise))
        for i in range(self.pop_zise): #calculate current fitness of all genotypes
            positions,target=environment.runTrial(self.pop[i])
            fitness_matrix[i]=fitness(positions,target)
        for gen in range(self.generations):
            if outputs:
                print("Generation",gen,"best fitness:",np.max(history))
            for i in range(self.pop_zise): #select three distinct individuals
                indices = [idx for idx in range(self.pop_zise) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                x1, x2, x3 = self.pop[r1].geno, self.pop[r2].geno, self.pop[r3].geno
                dummy=self.contr(*self.params) #create dummy object
                mutant = x1 + 0.2 * (x2 - x3)
                dummy.geno=mutant #set the geno to the object
                dummy.reform() #set the weights correct
                #crossover
                mutant=self.pop[i].sex(self.pop[i],dummy)
                #selection
                positions,target=environment.runTrial(mutant)
                trial_fitness=fitness(positions,target)
                if trial_fitness>fitness_matrix[i]:
                    fitness_matrix[i]=trial_fitness
                    self.pop[i]=deepcopy(mutant)
            history.append(np.max(history))
        return np.array(history), fitness_matrix
if __name__ == "__main__": #test code to run in same file
    _INPUT_SIZE_=10
    _H1_=10
    _H2_=10
    _OUTPUTSIZE_=10
    from GA_MODEL import controller, controllerCNN 
    def fitness_func(positions,target):
        return np.random.random((1))
    class environment_example:
        def __init__(self):
            pass
        def runTrial(self,geno):
            return [0,0,0,0],[0,0,0,0]

    ga=Differential(100,10,0.2,sex=1)
    ga.initialize_population(controller,[_INPUT_SIZE_,[_H1_,_H2_],_OUTPUTSIZE_])
    history,fitness=ga.evolve(environment_example(),fitness_func,outputs=1) 
    print(history)

    #cnn example
    ga=Differential(100,10,0.2,sex=1)
    
    input_shape = (20, 20)  #20x20 grayscale input image
    kernel_sizes = [[3,2], [3,2]]   #two convolution layers with 3x3 kernels
    hidden_size=512
    num_kernels = [4, 8]    #first layer has 4 kernels, second has 8 kernels
    output_size = 5         #output is 5 values (could be motor commands, classification, etc.)

    ga.initialize_population(controllerCNN, [input_shape, hidden_size, output_size, kernel_sizes, num_kernels])
    history,fitness=ga.evolve(environment_example(),fitness_func,outputs=1)
    print(history)

