import sys
sys.path.insert(1,"/its/home/drs25/Documents/GitHub/ant_trajectory") #put path here
from GA_code.GAs.GA_MODEL import *
from GA_code.GAs.genetic_algorithm import *
from grid_environment import environment
from RL_code.DQNAgent import DQNAgent

env=environment() #call in demo environment

agent=DQNAgent()


