

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import trajectory_code.trajectory_process_functions as tpf
import cv2

#%%

#calculate and plot a set of omniverse coordinates on top of an arena image

omni_food_coord = (0.15,-0.003,0.43)
food_radius = 3/100 # in metres
catch_radius = 23/100 #in metres

img_food_coord  = (542, 652)
panel_ends = [(147, 569), (301, 1012)]

#make "pixels per centimeter" ratio
ppm = tpf.distance(panel_ends[0], panel_ends[1])/0.4 

#how many metres away in the omniverse y positive direction (towrds start of arena) to grid
y_pos = 0.65
#how many metres away in the y omniverse negative direction (towrds end of arena) to grid
y_neg = -0.45
#how many metres away in the x positive direction (towrds longest arena edge) to grid
x_pos = 0.32
#how many metres away in the x negative direction (towards the very large cylinder cue outside areana)  to grid
x_neg = -0.50

#want each grid point to be 2.5 cm (0.025 metres) apart
x_spacing = int((x_pos-x_neg)/0.025)+1
y_spacing = int((y_pos-y_neg)/0.025)+1

#make the grid of values for inference
x_grid= omni_food_coord[0] + np.linspace(x_neg, x_pos, x_spacing)#[::2]
y_grid= omni_food_coord[1] + np.linspace(y_neg, y_pos, y_spacing)#[::2]


#create list of x and y values that form a grid
xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)
#flatten to single dimension array
xx_grid =  xx_grid.flatten()
yy_grid =  yy_grid.flatten()

plt.scatter(xx_grid, yy_grid)
plt.ylim(-0.5, 0.5)
plt.xlim(-0.5, 0.5)
plt.show()


xx_grid_pixels = img_food_coord[0] + (xx_grid-omni_food_coord[0])*ppm # order matters in python
yy_grid_pixels = img_food_coord[1] - (yy_grid-omni_food_coord[1])*ppm # axis offset for y

#%
figure, axes = plt.subplots()

#plot the grid for inference
plt.scatter( xx_grid_pixels, yy_grid_pixels, s=0.1)

figure.set_dpi(200)
#arena_img = plt.imread(r"\trajectory_code\top-down_arena.png")
arena_img = plt.imread(r"D:\Users\seyij\Projects\ant_trajectory\trajectory_code\top-down_arena.png")
plt.imshow(arena_img)
#make circles and add to plot
catch_circle = plt.Circle((img_food_coord[0], img_food_coord[1]), 
                          catch_radius*ppm , fill = False, color="green")
food_circle = plt.Circle((img_food_coord[0], img_food_coord[1]), 
                         food_radius*ppm , fill = False, color="red")
axes.add_artist(catch_circle)
axes.add_artist(food_circle)
plt.show()

print(x_grid)
print(y_grid)
#%% plot a single trajectory

traject_coords1 = np.load(r"D:\Users\seyij\Projects\trial2025-05-15 15_30_42.274644.npy")[3] # extract one trajectory

start_position = (0.08, 0.6) # start position in metres in the grid/ simulator

#extract x and y values
x_vals = traject_coords1[:, 0]
y_vals = traject_coords1[:, 1]

# get the difference between start x value and each trajectory value. 
x_diff = x_vals - start_position[0]
scale = 0.4
#Then apply a scaling factor to this difference before adding back to the starting x value.
x_vals = start_position[0] + (x_diff*scale)
#making y value increase instead of decrease in respect to origin point
#then takeaway one metre
y_vals = (start_position[1] + (start_position[1] - y_vals))-1


x_vals_pixels = ((x_vals-omni_food_coord[0])*ppm)+img_food_coord[0]
y_vals_pixels = ((y_vals-omni_food_coord[1])*ppm)+img_food_coord[1]


plt.imshow(arena_img)
plt.scatter(img_food_coord[0], img_food_coord[1], marker= "x")

plt.plot(x_vals_pixels, y_vals_pixels)
plt.show()

#%% loop to plot all

traject_coords = np.load(r"D:\Users\seyij\Projects\trial2025-05-15 15_30_42.274644.npy")
 # extract one trajectory
start_position = (0.08, 0.6)

plt.imshow(arena_img)
plt.scatter(img_food_coord[0], img_food_coord[1], marker= "x")
for traject in traject_coords:
#extract x and y values
    x_vals = traject[:, 0]
    y_vals = traject[:, 1]

    # get the difference between start x value and each trajectory value. 
    x_diff = x_vals - start_position[0]
    scale = 0.4
    #Then apply a scaling factor to this difference before adding back to the starting x value.
    x_vals = start_position[0] + (x_diff*scale)
    #making y value increase instead of decrease in respect to origin point
    y_vals = (start_position[1] + (start_position[1] - y_vals))-1

    x_vals_pixels = ((x_vals-omni_food_coord[0])*ppm)+img_food_coord[0]
    y_vals_pixels = ((y_vals-omni_food_coord[1])*ppm)+img_food_coord[1]
    
    plt.plot(x_vals_pixels, y_vals_pixels)
    
plt.show()

# function version of it in tpf

#%% using different image for calculating ratios

mark_image_path = r"D:\Users\seyij\Projects\ant_trajectory\trajectory_code\top_down_is_with_marks.png"
mark_image = plt.imread(mark_image_path)

omni_food_coord = (0.15,-0.003,0.43)
omni_start_coord = (0.08, 0.6, 0.43)
metre_distance = tpf.distance(omni_start_coord[:2], omni_food_coord[:2])

pixel_food_coord  = tpf.find_points(mark_image, "mark food location")[0]
pixel_start_coord = tpf.find_points(mark_image, "mark start coordinate")[0]
pixel_distance = tpf.distance(pixel_start_coord, pixel_food_coord)

#make "pixels per centimeter" ratio
ppm = pixel_distance/metre_distance

#%%

food_radius = 3/100 # in metres
catch_radius = 23/100 #in metres
#how many metres away in the omniverse y positive direction (towrds start of arena) to grid
y_pos = 0.65
#how many metres away in the y omniverse negative direction (towrds end of arena) to grid
y_neg = -0.45
#how many metres away in the x positive direction (towrds longest arena edge) to grid
x_pos = 0.32
#how many metres away in the x negative direction (towards the very large cylinder cue outside areana)  to grid
x_neg = -0.50

#want each grid point to be 2.5 cm (0.025 metres) apart
x_spacing = int((x_pos-x_neg)/0.025)+1
y_spacing = int((y_pos-y_neg)/0.025)+1

#make the grid of values for inference
x_grid= omni_food_coord[0] + np.linspace(x_neg, x_pos, x_spacing)#[::2]
y_grid= omni_food_coord[1] + np.linspace(y_neg, y_pos, y_spacing)#[::2]


#create list of x and y values that form a grid
xx_grid, yy_grid = np.meshgrid(x_grid,y_grid)
#flatten to single dimension array
xx_grid =  xx_grid.flatten()
yy_grid =  yy_grid.flatten()

plt.scatter(xx_grid, yy_grid)
plt.ylim(-0.5, 0.7)
plt.xlim(-0.5, 0.7)
plt.show()


xx_grid_pixels = pixel_food_coord[0] + (xx_grid-omni_food_coord[0])*ppm # order matters in python
yy_grid_pixels = pixel_food_coord[1] - (yy_grid-omni_food_coord[1])*ppm # axis offset for y

#%
figure, axes = plt.subplots()

#plot the grid for inference
plt.scatter( xx_grid_pixels, yy_grid_pixels, s=0.1)

figure.set_dpi(200)
#arena_img = plt.imread(r"\trajectory_code\top-down_arena.png")
arena_img = plt.imread(r"D:\Users\seyij\Projects\ant_trajectory\trajectory_code\top_down_is.png")
plt.imshow(arena_img)
#make circles and add to plot
catch_circle = plt.Circle((pixel_food_coord[0], pixel_food_coord[1]), 
                          catch_radius*ppm , fill = False, color="green")
food_circle = plt.Circle((pixel_food_coord[0], pixel_food_coord[1]), 
                         food_radius*ppm , fill = False, color="red")
axes.add_artist(catch_circle)
axes.add_artist(food_circle)
plt.show()

print(x_grid)
print(y_grid)
    

#%% loop to plot all with new image and ratio

traject_coords = np.load(r"D:\Users\seyij\Projects\trial2025-05-15 15_30_42.274644.npy")
 # extract one trajectory
start_position = (0.08, 0.6)

plt.imshow(arena_img)
plt.scatter(pixel_food_coord[0], pixel_food_coord[1], marker= "x")
for traject in traject_coords:
#extract x and y values
    x_vals = traject[:, 0]
    y_vals = traject[:, 1]

    # get the difference between start x value and each trajectory value. 
    x_diff = x_vals - start_position[0]
    scale = 0.4
    #Then apply a scaling factor to this difference before adding back to the starting x value.
    x_vals = start_position[0] + (x_diff*scale)
    #making y value increase instead of decrease in respect to origin point
    y_vals = (start_position[1] + (start_position[1] - y_vals))-1

    x_vals_pixels = ((x_vals-omni_food_coord[0])*ppm)+pixel_food_coord[0]
    y_vals_pixels = ((y_vals-omni_food_coord[1])*ppm)+pixel_food_coord[1]
    
    plt.plot(x_vals_pixels, y_vals_pixels)
    
plt.show()

# function version of it in tpf
#%% testing edits to plot

import trajectory_code.trajectory_process_functions as tpf

#traject_coords = np.load(r"D:\Users\seyij\Projects\trial2025-05-15 15_30_42.274644.npy")
traject_coords1 = np.load(r"D:\Users\seyij\Projects\ant_trajectory\data\trial\standard_trial19\routes.npy")

tpf.transform_model_trajects(traject_coords1, savefig = "test_offset.png", x_scale=1)







