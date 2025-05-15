# -*- coding: utf-8 -*-
"""
@author: oj74
"""

#read in a deeplabcut file, see what happens

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

#%

def distance(point1, point2):#points can be arrays or tuple or lists, as long as they have 2 items
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#extract a single frame from a video
def ext_frame(vid_name, frame_index=0, out_name=False):
    """ Extract a frame or multiple frames from a video.
    
    Keyword arguments:
        vid_name -- name of input video
        
        frame_index -- frame or frame number to be extracted. Accepts integer for single frame. Accepts tuple or list with 2 numbers representing range of frames to be extracted.
        
        out_name -- name of image file output for single frame, or folder output for multiple frames (default False). 
    
    Returns:
        If extracting single frame with out_name=False, the function returns image matrix, with a out_name it writes an image file in the current working directory. If extracting multiple frames it writes a folder of images or writes many images to the cwd. 
    
    """
    if type(frame_index) == int:
        vid = cv2.VideoCapture(vid_name)
        print("Total amount of frames: " + str(vid.get(7)))
        vid.set(1, frame_index)
        ret, frame = vid.read()
        vid.release()
        if out_name == False:
            #colour correction
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            cv2.imwrite(out_name, frame)
    #section for mutiple frames
    else:
        vid = cv2.VideoCapture(vid_name)
        a,b = frame_index
        if out_name == False:
            out_name = ""
            slash = ""
        else:
            os.mkdir(out_name)
            slash = "\\"
        #range cant iterate tuples, but it accepts multiple integers
        for number in range(a, b):
            vid.set(1, number)
            ret, frame = vid.read()
            cv2.imwrite(out_name+slash+str(number)+".png", frame)
        vid.release()
        
# calculate heading from dy/dx. Angles are relative to 0 degrees being "true north" of image
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

#function allowing you to click multiple points and it reports the pixel coordinate
# it will create a global variable name "coord", can be ignored.
#need to press escape to exit
def find_points(image, text = ""):
    #copy image so that original is not affected.
    my_image_wtext = image.copy()
    coord_list = []
    # add text to image 
    BLACK = (255,255,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.1
    font_color = BLACK
    font_thickness = 2
    #text
    x,y = 50,700
    cv2.putText(my_image_wtext, text+". Press space to continue", (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(my_image_wtext, "Press backspace to refresh", (x,y+50), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    #make image variable for after text has been added
    my_image_wdots = my_image_wtext.copy()
 
    def click_event(event, x, y, flags, params):
        #make a global variable
        global coord
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Coordinates: ", x, y)
            coord = (x, y)
            coord_list.append(coord)
            # put coordinates as text on the image
            cv2.putText(my_image_wdots, f'({x},{y})',(x,y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # 1 is fontScale
            
            # draw point on the image
            #3 is radius
            cv2.circle(my_image_wdots, (x,y),3, (0,255,255), -1)        

    while True:
        cv2.imshow("my_image_window", my_image_wdots)
        cv2.setMouseCallback("my_image_window", click_event)
        key = cv2.waitKey(1)  # Wait for a key event with a 1ms delay
        if key == 8: #check if the backspace has been pressed
        #reset the coords list and the image
            my_image_wdots = my_image_wtext.copy()
            coord_list = []
        if key == 32:  # Check if the pressed key is the Escape key (27 is the ASCII code for Escape)
            break
    cv2.destroyAllWindows()
    return coord_list
#%

def traject_process_for_omni(path_to_video, path_to_traject, out_plot_name, out_csv_name, plot_text=""):

    #read in a snapshot from the video file
    arena_img = ext_frame(path_to_video, frame_index=100)
    #arena_img = cv2.imread(r"C:\Users\oj74\Documents\oj_rotation3\pilot_analysis\testA_ant1_image.jpg")
    # want rgb version for matplotlib
    #arena_img= cv2.cvtColor(arena_img, cv2.COLOR_BGR2RGB)
    # read in video and and associated trajectory file from deeplabcut
    traject_df = pd.read_csv(path_to_traject, header=2)
    traject_df.columns = ["frame_number", "body_x", "body_y", "body_prob"]
    
    #% #plot the raw points from the tracking file using cv2
    arena_img_plot = arena_img.copy()
    x = traject_df["body_x"]
    y = traject_df["body_y"]
    #add points to the image
    for x, y in zip(x,y):
        cv2.circle(arena_img_plot, (int(x), int(y)), 2, (255,0,0), -1)
    #define properties and add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.1
    font_color = (255,255,255)
    font_thickness = 2
    text = ""
    cv2.putText(arena_img_plot, text+"Raw tracking path. Press space to continue", (50,700), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    #display the image through cv2
    while True:
        cv2.imshow("my_plot_window", arena_img_plot)
        key = cv2.waitKey(1)  # Wait for a key event with a 1ms delay
        if key == 32:  # Check if the pressed key is the Escape key (27 is the ASCII code for Escape)
            break
    cv2.destroyAllWindows()
    
    # get the fps of the video
    # Open the video file
    video = cv2.VideoCapture(path_to_video)
    # Get the FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    # Print the FPS
    print(f"Video Frames per second: {fps}")
    # Release the video capture object
    video.release()

    #% add time to dataframe
    traject_df["time"] = traject_df["frame_number"]* (1/fps)
    
            
    # add dy and dx to data frame
    dx_ser = traject_df["body_x"].diff(periods=1) #diff with periods =1 will takeaway the current row from the following row (x2-x1, y2- y1)
    dy_ser = traject_df["body_y"].diff(periods=1)
    heading_list= [] #initialise list ot store headings
    #calculate headings
    for (dx, dy) in zip(dx_ser, dy_ser):
        heading = heading_from_d(dx, dy)
        heading_list.append(heading)
        
    #% add headings to data frame
    traject_df["headings"] = heading_list
    
    #% return a list of coordinates for boundaries of the arena
    coord_list = find_points(arena_img, text= "mark the boudaries of the arena")
    # make a contour (cv2 shape) check whether points are inside the arena or not. They can later be filled in.
    arena_contour = np.array(coord_list, dtype=np.int32) 
    
    within_arena_list = []
    #cv2. pointpolygontest returns -1 if pint you give it is not in an arena
    for coord in zip(traject_df["body_x"], traject_df["body_y"]):
        result = cv2.pointPolygonTest(arena_contour, (coord), False) 
        within_arena_list.append(result)
    
    #add information on whether point is in arena to data the data frame
    traject_df["in_arena"] = within_arena_list
    
    #find ends of a panel which we know is 40cm
    panel_ends = find_points(arena_img, text= "mark both ends of a 40cm panel")
    #make "pixels per centimeter" ratio
    ppc = distance(panel_ends[0], panel_ends[1])/40
    
    #add positions in centimetres to data frame
    traject_df["body_x_cm"] = traject_df["body_x"]/ppc
    traject_df["body_y_cm"] = traject_df["body_y"]/ppc
    
    #get the origin point (entrance to the arena) through user input
    origin_point = find_points(arena_img, text= "mark the arena entrance point")[0] #[] because the function outputs a list
    
    #get every coordinate relative to the origin point in pixels
    traject_df["rel_x"] = traject_df["body_x"]-origin_point[0]
    traject_df["rel_y"] = traject_df["body_y"]-origin_point[1]
    #get the same in pixels
    traject_df["rel_x_cm"] = traject_df["rel_x"]/ppc
    traject_df["rel_y_cm"] = traject_df["rel_y"]/ppc
    
    # cut out uneeded items from the data
    #filter out items which are not in the arena area (as these are tracking errors)
    traject_df_cut = traject_df[traject_df["in_arena"]>-1]
    
    #cut out rows that have nan values 
    traject_df_cut.dropna(subset = ["headings"], inplace=True)
    
    #save the data frame to file as a csv
    traject_df_cut.to_csv(out_csv_name) #, index=False)
    print("csv saved as " + out_csv_name)
    
    #fill in nan values in the headings column with the last normal value observed
    #using .loc to replace all rows and columns in the dataframe so that i don't get pandas error
    #traject_df_cut.loc[:,:] = traject_df_cut.fillna(method="ffill")
    
    #get the location of the food with user input
    food_point = find_points(arena_img, text= "mark the food location point")[0]
    
    #% load data for plotting and plot quiver plot with arrows, then plot and save plot
    interval = 10 #interval is so that every 10th point is plotted, otherwise the graph looks too busy
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
    plt.scatter(food_point[0], food_point[1], s=6)
    plt.title("Heading plot (every 10th value) \n" + plot_text)
    #plt.show()
    plt.savefig(out_plot_name, bbox_inches='tight')
    print("plot saved as " + out_plot_name)

#produces plots showing route from accepted catchement area to successfully reaching the food
#also returns cut down csv with details from this first route
def get_success_plot(path_to_video, path_to_traject, out_plot_name, out_csv_name, plot_text=""):
    
    #% extract neccesary  data frame
    traject_df = pd.read_csv(path_to_traject)#, header=2)
    arena_img = ext_frame(path_to_video)
    # want rgb version for matplotlib
    #arena_img= cv2.cvtColor(arena_img, cv2.COLOR_BGR2RGB)
    #% plot
    
    x = traject_df["body_x"]
    y = traject_df["body_y"]
    
    #%check for correct colours
    plt.figure(dpi= 200)
    plt.imshow(arena_img)
    plt.scatter(x,y, s=0.1)
    plt.show()
    
    #% calculate food location and pixels per cm ration
    
    food_coord = (traject_df["body_x"][0] - traject_df["food_rel_x"][0], traject_df["body_y"][0] - traject_df["food_rel_y"][0])
    ppc = traject_df["body_y"][0]/traject_df["body_y_cm"][0]#extract pixels per cm ratio
    
    #% draw both circles
    
    food_radius = 3 #in centimetres
    catch_radius = 23 #in centimetres
    
    figure, axes = plt.subplots()
    #%check for correct colours
    figure.set_dpi(200)
    plt.imshow(arena_img)
    plt.scatter(x,y, s=0.1)
    #make circles and add to plot
    catch_circle = plt.Circle(food_coord, ppc*catch_radius , fill = False, color="green")
    food_circle = plt.Circle(food_coord, ppc*food_radius , fill = False, color="red")
    axes.add_artist(catch_circle)
    axes.add_artist(food_circle)
    plt.show()
    
    #% add distance from food to column so you can do indexing
    
    dist_to_food_list = []
    for coord in zip(traject_df["body_x"], traject_df["body_y"]):
         dist= distance(coord, food_coord)
         dist_to_food_list.append(dist/ppc) #get in cm
    
    #add to data frame
    traject_df["dist_to_food_cm"] = dist_to_food_list
    
    #% cut the data frame 
    
    #check if there are values that hit the food location
    if any(traject_df["dist_to_food_cm"] < food_radius) == True:
        print("reached food!!")
    else:
        print("didn't find food. so won't be processed")
        return
    
    
    #filter out value within catch radius AND first 5 seconds of video
    catchment_df = traject_df[(traject_df["dist_to_food_cm"] < catch_radius) & (traject_df["time"] > 5)]
    
    #%find the first instance where the ant reaches the food location withing the cathment data frames
    
    goal_index = catchment_df[catchment_df["dist_to_food_cm"] < 3].iloc[0].name
    
    #need to select up to the INDEX value, not just the first x values in the dataset
    first_bout_df = catchment_df.loc[:goal_index]
    
    #%save the csv
    first_bout_df.to_csv(out_csv_name, index=False)
    print("csv saved as " + out_csv_name)
    
    #% plot the first bout to target area
    interval = 10
    x = first_bout_df["body_x"][::interval]
    y = first_bout_df["body_y"][::interval]
    headings = first_bout_df["headings"][::interval]
    times = first_bout_df["time"][::interval]
    
    figure, axes = plt.subplots()
    figure.set_dpi(500)
    plt.imshow(arena_img)
    #make circles and add to plot
    catch_circle = plt.Circle(food_coord, ppc*catch_radius , fill = False, color="green")
    food_circle = plt.Circle(food_coord, ppc*food_radius , fill = False, color="red")
    axes.add_artist(catch_circle)
    axes.add_artist(food_circle)
    
    plt.quiver(
            x, y,
            # this parameter is x direction. Needs the angular offset
            np.sin(np.deg2rad(headings+90)),
            #this parameter is y direction
            np.cos(np.deg2rad(headings+90)),
            #the 5 argument is for the colours of the arrows
            #colour will be according to the time value
            times, ec="k")#,scale=80,width=0.01)
    
    plt.title(f"Heading plot (every {interval} values) \n" + plot_text)
    
    plt.savefig(out_plot_name, bbox_inches='tight')
    print("plot saved as " + out_plot_name)