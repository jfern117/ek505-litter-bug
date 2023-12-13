from ultralytics import YOLO
import numpy as np
import os
import pandas as pd


#using the pretrained yolo model from other data base
path_to_model = "..\\litter-detection\\runs\\detect\\train\\yolov8s_100epochs\\weights\\best.pt"

#create our yolo model
model = YOLO(path_to_model)

#create a list of all the files in the data directory
data_dir = "Litter Data"
data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

#Outline
#   - step 1: get path to every file
#   - step 2: for each file
#       - record name and ground truth info (boolean trash or not)
#       - run the model and record outputs (and save output image)
#   - step 3: export data to .csv (or similar) and begin analysis

list_of_trash = ["can", "bottle", "cardboard", "paper", "plastic"]
list_of_environments = ["dirt", "pavement", "grass"]

results = model(source=data_files, save=True)

# results = model(source="Litter Data\\can_dirt1.jpg", save = True)
# print(results[0].names[np.int32(results[0].boxes.cls[0])])
# print(np.float64(results[0].boxes.conf[0]))

result_data = []
columns = ["File_Name", "GT_contains_trash", "GT_trash_category", "GT_env_is_dirt", "GT_env_is_pavement", "GT_env_is_grass", "model_detects_trash", "model_top_detection_class", "model_top_detection_confidence"]
for idx in range(len(results)):
    curr_data = []
    result = results[idx]

    #get the file name
    curr_file_name = os.path.basename(data_files[idx])
    curr_data.append(curr_file_name)

    #save if file contains trash
    trash_label_check = [curr_category in curr_file_name.lower() for curr_category in list_of_trash]
    contains_trash = True if np.any(trash_label_check) else False
    curr_data.append(contains_trash)

    #save trash label
    curr_label = list_of_trash[np.argmax(trash_label_check)] if contains_trash else None
    curr_data.append(curr_label)

    #check if is dirt
    is_dirt = True if "dirt" in curr_file_name.lower() else False
    curr_data.append(is_dirt)

    #check if is pavement
    is_pavement = True if "pavement" in curr_file_name.lower() else False
    curr_data.append(is_pavement)

    #check if is grass
    is_grass = True if "grass" in curr_file_name.lower() else False
    curr_data.append(is_grass)

    #save if model detects trash
    detects_trash = True if len(result.boxes.cls) > 0 else False
    curr_data.append(detects_trash)

    #save top detection class
    detection_class = result.names[np.int32(result.boxes.cls[0])] if detects_trash else None
    curr_data.append(detection_class)

    #save top detection confidence
    detection_confidence = np.float64(result.boxes.conf[0]) if detects_trash else None
    curr_data.append(detection_confidence)

    #save the data
    result_data.append(curr_data)

data_frame = pd.DataFrame(result_data, columns = columns)
csv_file_name = "result_data.csv"
data_frame.to_csv(csv_file_name, index = False)



# print()
# print(result[0].boxes.cls)
# print(result[0].names[np.int32(result[0].boxes.cls[0])])
# print(result[0].boxes.conf)


#ouput
#File name #ground truth trash (yes/no contains trash) #model detects trash #