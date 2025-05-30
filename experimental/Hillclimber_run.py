import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from trajectory_code.trajectory_process_functions import transform_model_trajects
from GA_code.GAs.GA_MODEL import *
from GA_code.GAs.genetic_algorithm import *
from grid_environment import environment
import cv2
import matplotlib.pyplot as plt
import matplotlib
import datetime
import os
import pickle
def save_array_to_folder(base_dir, folder_name, fitness, pathways, ga,best_genotype,genes_time):
    folder_path = os.path.join(base_dir, folder_name)
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    # Save the array
    np.save(folder_path+"/fitnesses", fitness)
    
    transform_model_trajects(pathways, 
        image_path="/its/home/drs25/ant_trajectory/trajectory_code/testA_ant1_image.jpg", savefig=folder_path+"/pathsTaken.pdf", x_scale=1)
    transform_model_trajects(genes_time, 
        image_path="/its/home/drs25/ant_trajectory/trajectory_code/testA_ant1_image.jpg", savefig=folder_path+"/genes_time_paths.pdf", x_scale=1,time_element=True)
    max_v = max(arr.shape[0] for arr in pathways)
    padded_pathways = []
    for arr in pathways:
        pad_length = max_v - arr.shape[0]
        if pad_length > 0:
            padded = np.vstack([arr, np.zeros((pad_length, 2))])
        else:
            padded = arr
        padded_pathways.append(padded)
    routes = np.stack(padded_pathways)
    np.save(folder_path+"/routes", routes)
    with open(folder_path+"/GA", 'wb') as f:
        pickle.dump(ga, f)
    max_v = max(arr.shape[0] for arr in genes_time)
    padded_pathways = []
    for arr in pathways:
        pad_length = max_v - arr.shape[0]
        if pad_length > 0:
            padded = np.vstack([arr, np.zeros((pad_length, 2))])
        else:
            padded = arr
        padded_pathways.append(padded)
    routes = np.stack(padded_pathways)
    np.save(folder_path+"/routes_over_time", routes)
    env=environment(record=1,filename=folder_path+"/output.avi")
    path,dist=env.runTrial(best_genotype)
    env.out.release()


    
    print(f"Array saved to: {folder_path}")

def fitness_(trajectory, distances):
    # Reward reducing distance to the target
    improvement = distances[0] - distances[-1]
    improvement = max(improvement, 0)  # clip negative improvements
    if len(distances)<len(np.arange(0,1,0.01)): improvement=0
    print(">>",improvement)
    return improvement

def run(experiment_name,generations,population):
    env=environment(randomize_start=1) #call in demo environment
    #setup the network parameters so that the input is all the pixels values and output 2 velocities for motors
    _INPUT_SIZE_=cv2.resize(env.getObservation(), (8, 48), interpolation = cv2.INTER_AREA).shape[:2] #the 2 is for the target location, but realistically it will not know where the target is
    _OUTPUTSIZE_=3
    #microbial GA
    ga=Hillclimbers(generations,population,2,sex=1) #
    #ga.initialize_population(controllerCNN_LRF,[_INPUT_SIZE_,200,_OUTPUTSIZE_],std=2)
    ga.initialize_population(controller,[32,[10,5],_OUTPUTSIZE_])
    print("Begin tria6l")
    history,fitness=ga.evolve(env,fitness_,outputs=True) #run the GA
    print(fitness.shape)
    #from here you will want to find the best one from ga population from our fitness matrix
    best_genotype=ga.pop[np.argmax(fitness)]
    pathways=[]

    for i in range(len(ga.pop)):
        geno=ga.pop[i]
        path,dist=env.runTrial(geno)
        pathways.append(path)

    genes_over_time=ga.best_genos_time
    genes_time=[]
    for i in range(len(ga.best_genos_time)):
        geno=ga.best_genos_time[i]
        path,dist=env.runTrial(geno)
        genes_time.append(path)
    
    save_array_to_folder("/its/home/drs25/ant_trajectory/data/HC/", experiment_name, history, pathways, ga,best_genotype,genes_time)
    ##########
    
if __name__=="__main__":
    run("test",2,3)