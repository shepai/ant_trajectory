import sys
sys.path.insert(1,"/its/home/drs25/ant_trajectory") #put path here
from GA_code.GAs.GA_MODEL import *
from GA_code.GAs.genetic_algorithm import *
from grid_environment import environment
import cv2
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use('TkAgg')
env=environment() #call in demo environment
#setup the network parameters so that the input is all the pixels values and output 2 velocities for motors
_INPUT_SIZE_=cv2.resize(env.getObservation(), (8, 48), interpolation = cv2.INTER_AREA).shape[:2] #the 2 is for the target location, but realistically it will not know where the target is
_OUTPUTSIZE_=3

def fitness(trajectory, distances):
    # Reward reducing distance to the target
    improvement = distances[0] - distances[-1]
    improvement = max(improvement, 0)  # clip negative improvements
    print(improvement)
    return improvement

#microbial GA
ga=Microbial_GA(2,5,2,sex=1) #
ga.initialize_population(controllerCNN_LRF,[_INPUT_SIZE_,200,_OUTPUTSIZE_],std=2)
print("Begin trial")
history,fitness=ga.evolve(env,fitness,outputs=True) #run the GA
print(fitness.shape)
#from here you will want to find the best one from ga population from our fitness matrix
best_genotype=ga.pop[np.argmax(fitness)]
import cv2
plt.plot(history)
plt.xlabel("Generations")
plt.ylabel("Popoulation max fitness value")
plt.title("Generations vs Fitness")
plt.tight_layout()
#plt.savefig("/its/home/drs25/ant_trajectory/data/trial/trial"+str(datetime.datetime.now())+".pdf")
plt.close()
del env
env=environment(record=1)
pathways=[]
def real_to_pixel_coords(real_coords, ppm, real_ref, pixel_ref):
    x_shifted = (real_coords[:, 0] - real_ref[0]) * ppm
    y_shifted = (real_coords[:, 1] - real_ref[1]) * ppm
    
    px = x_shifted + pixel_ref[0]
    py = y_shifted + pixel_ref[1]
    
    return np.column_stack((px, py))
ppm = 1172.5106609323432
omni_food_coord = (0.15, -0.003)
img_food_coord = (542, 652)
for i in range(len(ga.pop)):
    geno=ga.pop[i]
    path,dist=env.runTrial(geno)
    pathways.append(path)
    pixels = real_to_pixel_coords(np.array(path), ppm, omni_food_coord[:2], img_food_coord)
    plt.plot(pixels[:,0],pixels[:,1])
plt.title("Pathways")
plt.imshow(cv2.imread("/its/home/drs25/ant_trajectory/trajectory_code/top-down_arena.png"))
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.tight_layout()
pixels = real_to_pixel_coords(np.array([env.target]), ppm, omni_food_coord[:2], img_food_coord)
plt.scatter(pixels[0][0],pixels[0][1], marker="x")
print("IMAGE FOOD",pixels)
plt.text(pixels[0][0],pixels[0][1]+0.01,"Target")
plt.savefig("/its/home/drs25/ant_trajectory/data/trial/positions"+str(datetime.datetime.now())+".pdf")

max_v = max(arr.shape[0] for arr in pathways)
padded_pathways = []
for arr in pathways:
    pad_length = max_v - arr.shape[0]
    if pad_length > 0:
        padded = np.vstack([arr, np.zeros((pad_length, 2))])
    else:
        padded = arr
    padded_pathways.append(padded)
pathways = np.stack(padded_pathways)
np.save("/its/home/drs25/ant_trajectory/data/trial/trial"+str(datetime.datetime.now()),pathways)
path,dist=env.runTrial(best_genotype)
env.out.release()

plt.show()