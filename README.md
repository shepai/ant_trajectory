
# ant_trajectory

## Environment
We put together an envionrment that uses a lookup table from a real envrionment the ants are in. The explanation on how these images are gathered is explained further in the Ant trajectory data section. The environment in ```/grid_environment``` uses little memory as there is no 3D rendering going on. It makes a good candidate for simplistic gyms. Our GA and RL example programs in ```/experimental```. Some code needs to be altered so the correct paths are being used.


## Genetic algorithm
THe code for the genetic algorithm found in /GA_code has the controller (GA_MODEL) and the optimizer (genetic_algorithm). The optimiser code improts the controller neural network, and evolves it using the microbial algorithm. 

To adapt this to work for your problem, you must modify the fitness function in genetic_algorithm.py and have an environment object which interfaces with your environment. This class you make will need a method '''runTrial''' which takes in the controller object. Use step to pass your input data in and recieve actions for the agent. 

```python

class env(Env):
    def runTrial(self,agent):
        self.reset()
        positions=[]
        target=(SELF.TARGET)
        for i in range(TOTAL_TIME):
            observation=self.getObservaion()
            action=agent.step(observation)
            self.act(action)
            positions.append(self.currentPosition)
        return positions,target
```

## RF learning


# Ant trajectory data
Referring to data in drop box https://sussex.app.box.com/folder/318276494249. And in the github repo "success_plots"
Go to Success plots. Within this you will find csvs with corresponding plots. These are subsets of trajectories taken out from longer recordings tracking the ant across the whole arena. These trajectories are cut to where the ant went within 3 centimetres of the food location. For context the experiment had 2 stages. Training: The ants were released and were watched as they freely explored to find the food location which had a drop of colourless glucose water. Test: The food was removed from the arena and the ant was free to explore and look for the food. This allowed us to see where the ant "remembered" where the food is essentially. Most of the data we have is from the ant test runs

### Folder structure
Referring to data in drop box https://sussex.app.box.com/folder/318276494249.  
traject_data_for_omni - This contains all video mp4s and corresponding deeplabcut tracking output for every test done.  
Dated folders (ie 25_07_2023) - These are videos captured on different days. The data was recorded by multiple people so apologies for the multiple naming schemes.  
traject_plots - This contains plots of tracking from every video, good for a quick look at which videos have data you might find interesting, but it still includes errors.
success_plots_csvs - This is a subset of videos taken where the ant reached a specified catchment area. Where all of these have the additional information added as listed in the interpretation section below.
success_plots_csvs\selected_ones - This is an even further subset of videos that I used to create the "ants eye views" in Isaac Sim replicator. These routes were chosen due to being a good variaety of data.

### How to interpret ant trajectory CSVs
These are from tracking with deeplabcut and additional information about timem, distance etc done on a per video basis. The i See the file: "\success_plots_csvs\2023-07-05_testA_ant1.csv" for example. Each column is a single frame.  

Frame number - This how many frames into the original video the current frame is. You can do frame_number/time to get the framerate (fps) of the original video.  
body_x - The the x coordinate of the ant in pixels.  
body_y - The the y coordinate of the ant in pixels.  
body_prob - Deeplabcut output on how accurate it thinks it is at labelling that frame.  
time - Time lapsed since the start of the video.  
headings - I calculated the heading of the ant using the change in coordinates. 0 degrees is "north" looking towards arena entrance. Because it waas caluclated on a frame-by-frame basis, I imagine an "angle-aware" smoothing filter would work great!  
in_arena - This was how I filtered some Deeplabuct errors out so if safe to ignore.  
body_x_cm and body_y_cm - Same coordinates butin centimetres (I did the conversion per video to account for different zoom levels)  
rel_x, rel_y - Coordinates relative to the ant release location (feel free to ignore as release point changes a lot between videos).  
food_rel_x, food_rel_y - Ant coordinates relative to the food locations. Use this as the food location is the same between videos. 

# Replicator output data
The folder is here https://sussex.app.box.com/file/1843980287568, you will need to unzip due to amount of images.
Replicator is the Isaac Sim tool used for synthetic data data generations.

### Folder structure
exp1_views - These are views generated from the routes tracked in Success_plots. Taking a virtual camera through the places that the ant went in the scene.  
grid_infer_view - These are views taken in spaced evenly apart across a 23cm square around the food location. These were orignally made to be used for inference/testing purposes.  
radial_routes - These are views going in straight lines from compass coordinates towards the food location. This was originally made to compare the more sinusoidal traning routes to a straight training route to see the difference.  
tests_and_visualisations - These are a set of videos generated from within the Isaac Sim environment. This helps gauge what the scene looks like.  
meta_data.csv(s) - These files explain the images found in the view folders. They give x y position, headings ect for every point.   

### How to interpret meta file csvs
Use this to interpret and find what images corresponds to what point in the simulation environment. Notice this is now in metres as omniverse/Isaac sim now measures things in metres. Each row corresponds to one image/observation.  

Route_name - Name of the route. Also is the name of the folder that the point/image is in.  
img_name - Name of the image file.  
x_m, y_m, z_m - Coordinates in simulation space where the image was taken in metres.  
heading - 0 degrees is "north" looking towards arena entrance.  
time - time from start of images being taken.  
og_time - the time in the original data set (take into account rows that have been filtered out to not be used)
og_frame_number - the frame number in the original videos where the data was taken from


