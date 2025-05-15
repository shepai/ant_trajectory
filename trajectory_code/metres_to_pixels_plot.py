

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import trajectory_process_functions as tpf
import cv2


#%

#calculate and plot a set of omniverse coordinates on top of an arena image

omni_food_coord = (0.15,-0.003,0.43)
food_radius = 3/100 # in metres
catch_radius = 23/100 #in metres

img_food_coord  = (542, 652)
panel_ends = [(147, 569), (301, 1012)]

#make "pixels per centimeter" ratio
ppm = tpf.distance(panel_ends[0], panel_ends[1])/0.4 # 0.4 metres is length of a panel in real life

#how many metres away in the y positive direction (towrds end of arena) to grid
y_pos = 0.31
#how many metres away in the y negative direction (towrds start of arena) to grid
y_neg = -0.44
#how many metres away in the x positive direction (towrds longest arena edge) to grid
x_pos = 0.23
#how many metres away in the x negative direction (towards the very large cylinder cue outside areana)  to grid
x_neg = -0.32

#want each grid point to be 2.5 cm (0.025 metres) apart
x_spacing = int((x_pos-x_neg)/0.025)+1
y_spacing = int((y_pos-y_neg)/0.025)+1

#make the grid of values for inference
x_grid= np.linspace(x_neg, x_pos, x_spacing)#[::2]
y_grid= np.linspace(y_neg, y_pos, y_spacing)#[::2]

#x_grid=np.linspace(-catch_radius, catch_radius, 23)#[::2]
#y_grid= np.linspace(-catch_radius, catch_radius, 23)#[::2]

#create list of x and y values that form a grid
xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)
#flatten to single dimension array and add the food coordinate value
xx_grid = omni_food_coord[0] + xx_grid.flatten()
yy_grid = omni_food_coord[1] + yy_grid.flatten()

xx_grid_pixels = ((xx_grid-omni_food_coord[0])*ppm)+img_food_coord[0]
yy_grid_pixels = ((yy_grid-omni_food_coord[1])*ppm)+img_food_coord[1]

#%
figure, axes = plt.subplots()

#plot the grid for inference
plt.scatter( xx_grid_pixels, yy_grid_pixels, s=0.1)

figure.set_dpi(200)
arena_img = plt.imread(r"\trajectory_code\top-down_arena.png")
plt.imshow(arena_img)

#make circles and add to plot
catch_circle = plt.Circle((img_food_coord[0], img_food_coord[1]), 
                          catch_radius*ppm , fill = False, color="green")
food_circle = plt.Circle((img_food_coord[0], img_food_coord[1]), 
                         food_radius*ppm , fill = False, color="red")
axes.add_artist(catch_circle)
axes.add_artist(food_circle)
plt.show()
