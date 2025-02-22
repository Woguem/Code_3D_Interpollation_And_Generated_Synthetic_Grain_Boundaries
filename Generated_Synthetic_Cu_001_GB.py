"""
Created on Thu Jan 30 13:07:03 2025

@author: Dr Yen Fred WOGUEM 

@description: This script generates synthetic images of grain boundaries : 
             grain boundaries with and without noise, noise and vaccum in a material structure.
"""




import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import math
from joblib import Parallel, delayed
import os
import json
import scipy.ndimage as ndi
from datetime import datetime


start_time = datetime.now()  # Démarrer le chronomètre



pixel_size = 0.5

dataset = np.loadtxt('Data_for_genereted_GB_images.txt', skiprows=1) # Download data set containing Nye tensor and the coordinates of atoms

#Positions of atoms
X = dataset[:, 0]
Y = dataset[:, 1]
Z = dataset[:, 2]

#Component of Nye tensor (A)
A11 = dataset[:,3]
A12 = dataset[:,4]
A13 = dataset[:,5]

A21 = dataset[:,6]
A22 = dataset[:,7]
A23 = dataset[:,8]

A31 = dataset[:,9]
A32 = dataset[:,10]
A33 = dataset[:,11]




# Image generation function with dislocations and/or noise


def generate_image(spacing_pixels, num_dislocations, noise_percentage, size_x, size_y, A23, X, Y, noise=True):
    
    A23_total = {}
     
    
    key_name = f"{num_dislocations}_{spacing_pixels}"
    
    A23_total[key_name] = np.zeros((size_x,size_y))

    for k in range(num_dislocations): 
        
        
        X_pixels = np.round(X / pixel_size).astype(int)  # Angstrom → pixel conversion
           
        Y_pixels = np.round(Y / pixel_size).astype(int)  # Angstrom → pixel conversion
            
        translated_X = X_pixels + k * spacing_pixels 
        
        translated_Y = Y_pixels  # Keep Y unchanged (you can also move it if necessary)
        
        # Delete out-of-box values while maintaining X-Y correspondence
        valid_indices = [i for i in range(len(translated_X)) if translated_X[i] < size_x]
        translated_X = [translated_X[i] for i in valid_indices]
        translated_Y = [translated_Y[i] for i in valid_indices]  # Identical filtering
        A23 = [A23[i] for i in valid_indices]  # Identical filtering
        
        
        shift = 20 # Shift the dislocations to the left so that the space between boundary 1 
                   # and the 1st dislocation is roughly equal to the space between the last dislocation and boundary 2.
                    
        translated_X = [x - shift for x in translated_X]
            
        
        A23_map = np.zeros((size_x,size_y)) # Initialize the matrix map containing the translated dislocation
        
        size = len(translated_X)
        
        for l in range(size):
            x_idx = int((translated_X[l]) % size_x)
            y_idx = int((translated_Y[l]) % size_y)
            A23_map[y_idx, x_idx]  =  max(A23_map[y_idx, x_idx], A23[l])
        
        # Add the map of offset dislocations to the total map
        
        A23_total[key_name] += A23_map
        
        # Add noise if necessary
        
    if noise:
        num_noise = int(size_x * size_y * noise_percentage)
        noise_x = np.random.randint(0, size_x, num_noise)
        noise_y = np.random.randint(0, size_y, num_noise)
        
        for i in range(num_noise):
            A23_total[key_name][noise_y[i], noise_x[i]] = np.random.normal(0, 0.04)
        
        
    return A23_total



# Generate images with specific proportions


size_x =  400   
size_y =  400   


num_images = 3000

spacing_pixels = np.random.randint(10, 200, num_images) # Array containing differents Sapcing between dislocations

num_dislocations = np.array([int(np.round(size_x / n)) for n in spacing_pixels]) # Array containing the number of 
                                                                                    # dislocations for each spacing 

noise_percentage = 0.05  # 5% of noise

#print(num_dislocations, spacing_pixels, np.shape(spacing_pixels))        
   

images = {} # Dictionary to collect all images

c1 = 0 # Counter
c2 = 0 # Counter
c3 = 0 # Counter
c4 = 0 # Counter

for n_i in range(1, num_images+1, 1):
    
    print(n_i)

    if n_i <= int(0.3*num_images) :  # Create 30% of images with grain boundary dislocations and no other defects
        c1 += 1
        A23_total = generate_image(
            spacing_pixels=spacing_pixels[c1], 
            num_dislocations=num_dislocations[c1], 
            noise_percentage=noise_percentage,
            size_x=size_x, 
            size_y=size_y, 
            A23=A23, 
            X=X, 
            Y=Y, 
            noise=False)
        
        for key_1 in A23_total.keys():
            image = A23_total[key_1]
            key_name_1 = f"{key_1}_GB_system_{c1}"
            images[key_name_1] = image  # Add image with key
        
        print('GB')    
            
    elif int(0.3*num_images) < n_i <= int(0.6*num_images) :  # Create 30% of images without defects (vacuum)
        c2 += 1 
        A23_total = np.zeros((size_x, size_y))  # Blank image
        key_name = f"{0}_{0}_vacuum_system_{c2}"
        images[key_name] = A23_total  # Add image with key
        
        print('Vaccum')
        
    elif int(0.6*num_images) < n_i <= int(0.8*num_images):  # Create 20% of images with grain boundary dislocations 
                                                            # and with noise
        c3 += 1
        A23_total = generate_image(
            spacing_pixels=spacing_pixels[c3], 
            num_dislocations=num_dislocations[c3], 
            noise_percentage=noise_percentage, 
            size_x=size_x, 
            size_y=size_y, 
            A23=A23, 
            X=X, 
            Y=Y, 
            noise=True)
        for key_2 in A23_total.keys():
            image = A23_total[key_2]
            key_name_2 = f"{key_2}_GB_noise_system_{c3}"
            images[key_name_2] = image  # Add image with key
            
        
        print('GB_noise')   
        
    else:  # Create 20% of images only with noise 
        c4 += 1
        A23_total = np.zeros((size_x, size_y))  # Blank image
        num_noise = int(size_x * size_y * noise_percentage)
        noise_x = np.random.randint(0, size_x, num_noise)
        noise_y = np.random.randint(0, size_y, num_noise)
        for i in range(num_noise):
            A23_total[noise_y[i], noise_x[i]] = np.random.normal(0, 0.06) # Replacing vacuum with noise
        key_name = f"{0}_{0}_noise_system_{c4}"
        images[key_name] = A23_total # Add image with key
        
        print('noise')
    
        



def extract_dislocation_cores(images, save_dir, threshold=0.4): 
    """
    Detects all pixels with maximum intensity in each image.
    If the maximum intensity is below the threshold, the image is empty.

    Args:
        images (dict): Dictionary containing generated images { "key_name": np.array }.
        save_dir (str): Directory where JSON files will be saved.
        threshold (float): Minimum threshold for an image to be considered as containing dislocations.
        

    Returns:
            None (writes a JSON file containing annotations).
    """
    annotations = {}
    
    max_dislocation = 0

    for i, image in enumerate(images.values()):
        dislocations = []

        # Find the maximum image intensity
        max_intensity = np.max(image)
        
        #print(max_intensity)


        # If the maximum intensity is below the threshold, consider the image as empty 
                # (because there are vacuum images and images with only noise)
        if max_intensity < threshold:
            annotations[f"Image_{i}.png"] = {"dislocations": [{"x": 0, "y": 0, "p1": 0, "p0": 1}]}
            
        else :

            # Detection of pixels with intensity greater than 0.4 (first tens of max_intensity)
            local_maxima =  image >= threshold                                           
        
            # Grouping connected maxima into regions
            labeled_maxima, num_features = ndi.label(local_maxima)
        
            if num_features > max_dislocation : # Sert à récuperer le nombre maximale de dislocations qu'il peut y avoir dans une image
            
                max_dislocation = num_features

            # Find the center of each detected region using the barycenter
            centers = ndi.center_of_mass(image, labeled_maxima, range(1, num_features + 1))
            centers = np.round(centers).astype(int)  # Convert to integers
            # Adding centers detected as dislocations
            for y, x in centers:
                dislocations.append({"x": int(x), "y": int(y), "p1": 1, "p0": 0})

            # Save image annotations
            annotations[f"Image_{i}.png"] = {"dislocations": dislocations}

    
    # Create a complete path for the output JSON file
    output_file_1 = os.path.join(save_dir, 'Max_Dislocations.json')
    
    # Save to JSON file in specified location
    with open(output_file_1, "w") as f:
        json.dump({"max_dislocations": max_dislocation}, f, indent=4)

    print(f"Max dislocations saved in {output_file_1}")
    
    # Create a complete path for the output JSON file
    output_file = os.path.join(save_dir, 'Grain_Boundary.json')
    
    # Save to JSON file in specified location
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Labels saved")


# Define the directory in which to save images and the JSON file

save_dir = r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Generated_Images"   

# Create folder if none exists
os.makedirs(save_dir, exist_ok=True) 

extract_dislocation_cores(images, save_dir)  # Label creation




compt_image = -1
    
# Display images generated with different configurations
for key in images.keys():
    
    compt_image += 1
    
    # Create a figure for each image
    fig, ax = plt.subplots(figsize=(13.5, 9), dpi=300)
    
    # Calculating limits
    #extent_angstrong = [0, size_x * pixel_size, 0, size_y * pixel_size]
    
    extent = [0, size_x, 0, size_y]
    
    # Image display with “jet” color map, range of values defined between v_min and v_max
    img = ax.imshow(images[key], cmap='jet', origin='lower', vmin=-0.3, vmax=0.3, extent=extent)
    
    # Add a color bar to the right of the image
    #cbar = fig.colorbar(img, pad=0.1, location='right')
    #cbar.set_label(r'$\alpha_{23}$ [$\mathrm{\AA}^{-1}$]', fontsize=40) 
    #cbar.ax.yaxis.set_label_position('left')
    #cbar.ax.tick_params(labelsize=34)
    
    # Add labels and titles to axes
    #ax.set_xlabel(r'X [$\mathrm{\AA}$]', fontsize=34)
    #ax.set_ylabel(r'Y [$\mathrm{\AA}$]', fontsize=34)
    #ax.tick_params(axis='both', which='major', labelsize=34)
    
    ax.set_axis_off() # Disable axis display
    
    # Add image title
    #ax.set_title(f'Number_of_dislocations_and_Spacing_{key}', fontsize=14, fontweight='bold')
    
    # Save image as PNG
    save_path = os.path.join(save_dir, f'Image_{compt_image}.png')
    
    plt.savefig(save_path, dpi=300)
    
    
    # Display image in viewer window
    #plt.show()
    
    # Close all figures after display
    plt.close() 

    

end_time = datetime.now()  # Fin du chronomètre
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")





































