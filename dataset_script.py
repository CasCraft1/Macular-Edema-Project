import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import re


#Please note, this file uses local paths. Dataset is far too large to store directly onto github. 
#From this point forward, relative paths will be used operating on generated datasets


os.chdir(r'F:\Data\OLIVES\TREX_DME\TREX DME')
data = np.load(r"F:\Data\OLIVES\TREX_DME\TREX DME\TREX\0201TOS\V1\OS\volume_35_1.npy")

#check that data is images
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
print("Unique values:", np.unique(data))


#visualize first image
plt.imshow(data[0], cmap="gray")
plt.title("Image 0")
plt.colorbar()
plt.show()

#load in labels for data 
labels = pd.read_csv(r"F:\Data\Macular-Edema-Project\olives_visit_level_clean.csv")


labels = labels[labels['Regimen_Arm'] != 'PRN (PRIME)'] #remove non trex data for now


n = len(labels)
new_columns = list(labels.columns) + ['Image']
dataset = pd.DataFrame(columns= new_columns)
total_images = 0
for row in range(n):
    
    start = r'F:\Data\OLIVES\TREX_DME\TREX DME'
    regimen = labels['Regimen_Arm'][row]
    subject = labels['Subject_Folder'][row]
    laterality = labels['Laterality'][row]
    visit = labels['Visit_Label'][row]
    path = f'F:/Data/OLIVES/TREX_DME/TREX DME/{regimen}/{subject}/{visit}/{laterality}'

    os.chdir(path)
    volume_file = [file for file in os.listdir() if bool(re.search('.npy',file)) ][0]
    volume_path = f'{path}/{volume_file}'
    print(volume_file)
    volume = np.load(volume_path)

    #create row for each image and give it labels for the volume
    for index in range(volume.shape[0]):
        image = volume[index]
        image_label = labels.iloc[row]
        new_row = list(image_label) + [image]
        dataset.loc[total_images] = new_row
        total_images += 1


dataset.to_csv(r'F:\Data\Macular-Edema-Project\dataset.csv') # for a quick glance
dataset.to_pickle(r'F:\Data\Macular-Edema-Project\dataset.pk1') # to keep numpy arrays stored


