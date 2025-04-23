# ant_trajectory

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

# Ant trajectory data

Go to Success plots. Within this you will find csvs with corresponding plots. These are subsets of trajectories taken out from longer recordings tracking the ant across the whole arena. These trajectories are cut to where the ant went within 3 centimetres of the food location. For context the experiment had 2 stages. Training: The ants were released and were watched as they freely explored to find the food location which had a drop of colourless glucose water. Test: The food was removed from the arena and the ant was free to explore and look for the food. This allowed us to see where the ant "remembered" where the food is essentially. Most of the data we have is from the ant test runs

## How to interpret CSVs
The are from tracking with deeplabcut. See the file: "\success_plots_csvs\2023-07-05_testA_ant1.csv" for example. Each column is a single frame.  

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
