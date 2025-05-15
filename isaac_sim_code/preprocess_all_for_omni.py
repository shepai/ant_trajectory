# -*- coding: utf-8 -*-
"""
@author: oj74
"""

#script to preprocess all the video and trajectory data, ready to go into the omniverse

from trajectory_process_functions import traject_process_for_omni
from trajectory_process_functions import find_points
from trajectory_process_functions import ext_frame
from trajectory_process_functions import get_success_plot
import os
import pandas as pd


#%% get names of all folders with video recording data in them
db_path = r"C:\Users\oj74\OneDrive - University of Sussex\ant lab cloud\experiment_recordings"
all_folders = os.listdir(db_path)
db_suffix = "2023"
#get only the dated folders
video_folder_list = [os.path.join(db_path, folder) for folder in all_folders if folder.endswith(db_suffix)]

output_folder = r"C:\Users\oj74\Documents\oj_rotation3\traject_data_for_omni"

#%% per folder, get names of and process all files

for folder_path in video_folder_list[1:]:
    suffix = "filtered.csv"  # this will get all of the filtered deeplabcut files
    vid_suffix = ".mp4" # this will filter out a video
    
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter the list to include only files with the specified suffix
    filtered_csvs = [file for file in all_files if file.endswith(suffix)]
    
    #get list of videos. Making filtering for mp4 and ensuring that it doesn't pick up a DLC labelled video accidentally
    video_list = [file for file in all_files if file.endswith(vid_suffix) and "DLC" not in file]
    
    #iterate through list of videos with index to get corresponding file from the csv list
    for i, video_name in enumerate(video_list):
        csv_name = filtered_csvs[i]
        
        #make full path
        path_to_video = os.path.join(folder_path, video_name)
        path_to_traject = os.path.join(folder_path, csv_name)
        
        #choose output names
        out_plot_name = os.path.join(output_folder, video_name[:-4] +".png") #will be image file out
        out_csv_name = os.path.join(output_folder, video_name[:-4] + ".csv") # using video name as a base because it is shorter
        
        traject_process_for_omni(path_to_video, path_to_traject, out_plot_name, out_csv_name, plot_text=video_name[:-4])

#%% need to add relative positions to the food locations to the csvs

db_path = r"C:/Users/oj74/Documents/oj_rotation3/traject_data_for_omni"
# get names of all folders with video recording data in them
all_folders = os.listdir(db_path)
db_date = "2023"
#get only the dated folders
video_folder_list = [os.path.join(db_path, folder) for folder in all_folders if db_date in folder]

#%%


for folder_path in video_folder_list:
    suffix = ".csv"  # this will get all of the filtered deeplabcut files
    vid_suffix = ".mp4" # this will filter out a video
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter the list to include only files with the specified suffix
    csv_list = [file for file in all_files if file.endswith(suffix)]
    
    #get list of videos. Making filtering for mp4 and ensuring that it doesn't pick up a DLC labelled video accidentally
    video_list = [file for file in all_files if file.endswith(vid_suffix)]
    
    #extract frame from any video  in the folder
    arena_img = ext_frame(os.path.join(folder_path, video_list[1]), 100)
    #fin the food point
    food_coord = find_points(arena_img, "mark food location")[0]
    
    #process csvs in each folder
    for csv_name in csv_list:
        
        #make full path to trajectory
        path_to_traject = os.path.join(folder_path, csv_name)
        #load data frame
        traject_df = pd.read_csv(path_to_traject)
        
        if 'body_x' in traject_df.columns:
            print("body x is present and accounted for")
            #calculate pixels per cm
            ppc = traject_df["body_y"].iloc[1]/traject_df["body_y_cm"].iloc[1]
            
            #add relative to food location information
            traject_df["food_rel_x"] = traject_df["body_x"]-food_coord[0]
            traject_df["food_rel_y"] = traject_df["body_y"]-food_coord[1]
            #get the same in pixels
            traject_df["food_rel_x_cm"] = traject_df["food_rel_x"]/ppc
            traject_df["food_rel_y_cm"] = traject_df["food_rel_y"]/ppc
        
            #choose output names
            out_csv_name = os.path.join(folder_path, csv_name) # using video name as a base because it is shorter
            #save as csv without index
            traject_df.to_csv(out_csv_name, index=False)
            print("saved the file " + out_csv_name)
    
        else:
            print(f"'body_x' column not found in {csv_name}")

#%% get out all the successful routes and save the subset of success ones with plots in a new folder

db_path = r"C:\Users\oj74\Documents\oj_rotation3\traject_data_for_omni"
# get names of all folders with video recording data in them
all_folders = os.listdir(db_path)
db_date = "2023"
#get only the dated folders
video_folder_list = [os.path.join(db_path, folder) for folder in all_folders if db_date in folder]

output_folder = r"C:\Users\oj74\Documents\oj_rotation3\traject_data_for_omni\success_plots_csvs"

#%% per folder, get names of and process all files

for folder_path in video_folder_list[1:]:
    suffix = ".csv"  # this will get all of the filtered deeplabcut files
    vid_suffix = ".mp4" # this will filter out a video
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter the list to include only files with the specified suffix
    csv_list = [file for file in all_files if file.endswith(suffix)]
    
    #get list of videos. Making filtering for mp4 and ensuring that it doesn't pick up a DLC labelled video accidentally
    video_list = [file for file in all_files if file.endswith(vid_suffix) and "DLC" not in file]
    
    #iterate through list of videos with index to get corresponding file from the csv list
    for video_name, csv_name in zip(video_list, csv_list):
        
        #make full path
        path_to_video = os.path.join(folder_path, video_name)
        path_to_traject = os.path.join(folder_path, csv_name)
        
        #choose output names
        out_plot_name = os.path.join(output_folder, video_name[:-4] +".png") #will be image file out
        out_csv_name = os.path.join(output_folder, video_name[:-4] + ".csv") # using video name as a base because it is shorter
        
        get_success_plot(path_to_video, path_to_traject, out_plot_name, out_csv_name, plot_text=video_name[:-4])

