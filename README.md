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