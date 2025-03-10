# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:33:58 2025

@author: oj74
"""

#read in a deeplabcut file, see what happens

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
import numpy as np

def distance(point1, point2):#points can be arrays or tuple or lists, as long as they have 2 items
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#%% read in video and and assoiciated trajectory file from deeplabcut

path_to_video = r"C:\Users\oj74\Documents\oj_rotation3\pilot_analysis\testA_ant1.mp4"
path_to_traject = r"C:\Users\oj74\Documents\oj_rotation3\pilot_analysis\testA_ant1_trajectory_filtered.csv"

#read in an image too
arena_img = cv2.imread(r"C:\Users\oj74\Documents\oj_rotation3\pilot_analysis\testA_ant1_image.jpg")
# want rgb version for matplotlib
arena_img= cv2.cvtColor(arena_img, cv2.COLOR_BGR2RGB)

traject_df = pd.read_csv(path_to_traject, header=2)
traject_df.columns = ["frame_number", "body_x", "body_y", "body_prob"]

#%% get the fps of the video
# Open the video file
video = cv2.VideoCapture(path_to_video)
# Get the FPS
fps = video.get(cv2.CAP_PROP_FPS)
# Print the FPS
print(f"Frames per second: {fps}")
# Release the video capture object
video.release()

#%% add time to dataframe
traject_df["time"] = traject_df["frame_number"]* (1/fps)

#%%make heading from dy/dx function

def heading_from_d(dy, dx):
    
    #in case one of the values comes out as 0
    if dx ==0 and dy == 0:
        heading = np.nan
        
    elif dx == 0 and dy > 0:
        heading = 0
        
    elif dx == 0 and dy < 0:
        heading = 180
        
    elif dx < 0 and dy == 0:
        heading = 270
        
    elif dx > 0 and dy == 0:
        heading = 90
            
    else:
        #arctan2 does things in a quadrant sensitive way but scales things from -180 to 180.
        #arctan2 also measure angle from the x axis, so you need to offset by 90
        angle = np.rad2deg(np.arctan2(dy, dx))
        #need to use modulus % to scale things back to 0 to 360 degrees
        heading = (90-angle) % 360
    
    return heading
        
#%% add to data frame
#diff with periods =1 will takeaway the current row from the following row (x2-x1, y2- y1)
dx_ser = traject_df["body_x"].diff(periods=1)
dy_ser = traject_df["body_y"].diff(periods=1)

heading_list= []

for (dx, dy) in zip(dx_ser, dy_ser):
    heading = heading_from_d(dx, dy)
    heading_list.append(heading)
    
#% add to data frame
traject_df["headings"] = heading_list

#%%
#function allowing you to click multiple points and it reports the pixel coordinate
# it will create a global variable name "coord", can be ignored.
#need to press escape to exit
def find_points(image, text = False):
    #copy image so that original is not affected.
    my_image = image.copy()
    coord_list = []
    # add text to image 
    if text != False:
        BLACK = (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.1
        font_color = BLACK
        font_thickness = 2
        text = text
        x,y = 10,700
        cv2.putText(my_image, text+". Press esc to exit", (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
 
    def click_event(event, x, y, flags, params):
        #make a global variable
        global coord
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Coordinates: ", x, y)
            coord = (x, y)
            coord_list.append(coord)
            # put coordinates as text on the image
            cv2.putText(my_image, f'({x},{y})',(x,y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # 1 is fontScale
            
            # draw point on the image
            #3 is radius
            cv2.circle(my_image, (x,y),3, (0,255,255), -1)        

    while True:
        cv2.imshow("my_image_window", my_image)
        cv2.setMouseCallback("my_image_window", click_event)
        key = cv2.waitKey(1)  # Wait for a key event with a 1ms delay
        if key == 27:  # Check if the pressed key is the Escape key (27 is the ASCII code for Escape)
            break
    cv2.destroyAllWindows()
    return coord_list

#%% return a list of coordinates for boundaries of the arena
coord_list = find_points(arena_img, text= "mark the boudaries of the arena")
#%% make a contour (cv2 shape) check whether points are inside the arena or not. They can later be filled in.
arena_contour = np.array(coord_list, dtype=np.int32) 

within_arena_list = []

#cv2. pointpolygontest returns -1 if pint you give it is not in an arena
for coord in zip(traject_df["body_x"], traject_df["body_y"]):
    result = cv2.pointPolygonTest(arena_contour, (coord), False) 
    within_arena_list.append(result)

#%% add information on whether point is in arena to data frame and filter

traject_df["in_arena"] = within_arena_list
traject_df_cut = traject_df[traject_df["in_arena"]>-1]
#%% load data for plotting and plot quiver plot with arrows

#interval is so that every 10th point is plotted, otherwise the graph looks too busy
interval = 50

#get data from dataframe into easier to read variables
x = traject_df_cut["body_x"][::interval]
y = traject_df_cut["body_y"][::interval]
headings = traject_df_cut["headings"][::interval]


plt.figure(dpi= 500)
plt.imshow(arena_img)
plt.quiver(
        x, y,
        # this parameter is x direction. Needs the angular offset
        np.sin(np.deg2rad(headings+90)),
        #this parameter is y direction
        np.cos(np.deg2rad(headings+90)),
        #the 5 argument is for the colours of the arrows
        ec="k")#,scale=80,width=0.01)
plt.show()



