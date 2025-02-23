"""
@author: Dr Yen Fred WOGUEM & Dr Houssam KHAROUJI

@description: This script interpolates the crystal structure on a grid using the G-method.
    
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time
import numba as nb
from scipy.spatial import Voronoi, voronoi_plot_2d
import itertools
import math
from joblib import Parallel, delayed
from datetime import datetime


start_time = datetime.now()  # Start timer



pixel_size =  0.5

Gij = np.loadtxt('G_Cu_001_365.dump', skiprows=9)

#Positions of atoms
Xvec = Gij[:, 2]
Yvec = Gij[:, 3]
Zvec = Gij[:, 4]

#Component of Nye tensor (A)
G11 = Gij[:, 5]
G12 = Gij[:, 6]
G13 = Gij[:, 7]
G21 = Gij[:, 8]
G22 = Gij[:, 9]
G23 = Gij[:, 10]
G31 = Gij[:, 11]
G32 = Gij[:, 12]
G33 = Gij[:, 13]




#Positions Normalization first time
Xvec = Xvec - min(Xvec)
Yvec = Yvec - min(Yvec)
Zvec = Zvec - min(Zvec)

#System size
Xsize = max(Xvec)-min(Xvec)
Ysize = max(Yvec)-min(Yvec)
Zsize = max(Zvec)-min(Zvec)


cutoff = 4


# Find the positions of ghost atoms

# Right side ghost atoms
X_ghost_atoms_x_r = Xvec[np.where(Xvec <= cutoff)] + Xsize
Y_ghost_atoms_x_r = Yvec[np.where(Xvec <= cutoff)] 
Z_ghost_atoms_x_r = Zvec[np.where(Xvec <= cutoff)]
G11_ghost_atoms_x_r = G11[np.where(Xvec <= cutoff)]
G12_ghost_atoms_x_r = G12[np.where(Xvec <= cutoff)]
G13_ghost_atoms_x_r = G13[np.where(Xvec <= cutoff)] 
G21_ghost_atoms_x_r = G21[np.where(Xvec <= cutoff)]
G22_ghost_atoms_x_r = G22[np.where(Xvec <= cutoff)]
G23_ghost_atoms_x_r = G23[np.where(Xvec <= cutoff)]
G31_ghost_atoms_x_r = G31[np.where(Xvec <= cutoff)]
G32_ghost_atoms_x_r = G32[np.where(Xvec <= cutoff)]
G33_ghost_atoms_x_r = G33[np.where(Xvec <= cutoff)]

Y_ghost_atoms_y_r = Yvec[np.where(Yvec <= cutoff)] + Ysize
X_ghost_atoms_y_r = Xvec[np.where(Yvec <= cutoff)] 
Z_ghost_atoms_y_r = Zvec[np.where(Yvec <= cutoff)] 
G11_ghost_atoms_y_r = G11[np.where(Yvec <= cutoff)]
G12_ghost_atoms_y_r = G12[np.where(Yvec <= cutoff)]
G13_ghost_atoms_y_r = G13[np.where(Yvec <= cutoff)]
G21_ghost_atoms_y_r = G21[np.where(Yvec <= cutoff)]
G22_ghost_atoms_y_r = G22[np.where(Yvec <= cutoff)]
G23_ghost_atoms_y_r = G23[np.where(Yvec <= cutoff)]
G31_ghost_atoms_y_r = G31[np.where(Yvec <= cutoff)]
G32_ghost_atoms_y_r = G32[np.where(Yvec <= cutoff)]
G33_ghost_atoms_y_r = G33[np.where(Yvec <= cutoff)]

Z_ghost_atoms_z_r = Zvec[np.where(Zvec <= cutoff)] + Zsize
X_ghost_atoms_z_r = Xvec[np.where(Zvec <= cutoff)] 
Y_ghost_atoms_z_r = Yvec[np.where(Zvec <= cutoff)]
G11_ghost_atoms_z_r = G11[np.where(Zvec <= cutoff)]
G12_ghost_atoms_z_r = G12[np.where(Zvec <= cutoff)]
G13_ghost_atoms_z_r = G13[np.where(Zvec <= cutoff)]
G21_ghost_atoms_z_r = G21[np.where(Zvec <= cutoff)]
G22_ghost_atoms_z_r = G22[np.where(Zvec <= cutoff)]
G23_ghost_atoms_z_r = G23[np.where(Zvec <= cutoff)]
G31_ghost_atoms_z_r = G31[np.where(Zvec <= cutoff)]
G32_ghost_atoms_z_r = G32[np.where(Zvec <= cutoff)]
G33_ghost_atoms_z_r = G33[np.where(Zvec <= cutoff)]

# Left side ghost atoms
X_ghost_atoms_x_l = Xvec[np.where(Xvec <= cutoff)] - cutoff
Y_ghost_atoms_x_l = Yvec[np.where(Xvec <= cutoff)] 
Z_ghost_atoms_x_l = Zvec[np.where(Xvec <= cutoff)] 
G11_ghost_atoms_x_l = G11[np.where(Xvec <= cutoff)]
G12_ghost_atoms_x_l = G12[np.where(Xvec <= cutoff)]
G13_ghost_atoms_x_l = G13[np.where(Xvec <= cutoff)] 
G21_ghost_atoms_x_l = G21[np.where(Xvec <= cutoff)]
G22_ghost_atoms_x_l = G22[np.where(Xvec <= cutoff)]
G23_ghost_atoms_x_l = G23[np.where(Xvec <= cutoff)]
G31_ghost_atoms_x_l = G31[np.where(Xvec <= cutoff)]
G32_ghost_atoms_x_l = G32[np.where(Xvec <= cutoff)]
G33_ghost_atoms_x_l = G33[np.where(Xvec <= cutoff)]

Y_ghost_atoms_y_l = Yvec[np.where(Yvec <= cutoff)] - cutoff
X_ghost_atoms_y_l = Xvec[np.where(Yvec <= cutoff)] 
Z_ghost_atoms_y_l = Zvec[np.where(Yvec <= cutoff)] 
G11_ghost_atoms_y_l = G11[np.where(Yvec <= cutoff)]
G12_ghost_atoms_y_l = G12[np.where(Yvec <= cutoff)]
G13_ghost_atoms_y_l = G13[np.where(Yvec <= cutoff)]
G21_ghost_atoms_y_l = G21[np.where(Yvec <= cutoff)]
G22_ghost_atoms_y_l = G22[np.where(Yvec <= cutoff)]
G23_ghost_atoms_y_l = G23[np.where(Yvec <= cutoff)]
G31_ghost_atoms_y_l = G31[np.where(Yvec <= cutoff)]
G32_ghost_atoms_y_l = G32[np.where(Yvec <= cutoff)]
G33_ghost_atoms_y_l = G33[np.where(Yvec <= cutoff)]

Z_ghost_atoms_z_l = Zvec[np.where(Zvec <= cutoff)] - cutoff
X_ghost_atoms_z_l = Xvec[np.where(Zvec <= cutoff)] 
Y_ghost_atoms_z_l = Yvec[np.where(Zvec <= cutoff)] 
G11_ghost_atoms_z_l = G11[np.where(Zvec <= cutoff)]
G12_ghost_atoms_z_l = G12[np.where(Zvec <= cutoff)]
G13_ghost_atoms_z_l = G13[np.where(Zvec <= cutoff)]
G21_ghost_atoms_z_l = G21[np.where(Zvec <= cutoff)]
G22_ghost_atoms_z_l = G22[np.where(Zvec <= cutoff)]
G23_ghost_atoms_z_l = G23[np.where(Zvec <= cutoff)]
G31_ghost_atoms_z_l = G31[np.where(Zvec <= cutoff)]
G32_ghost_atoms_z_l = G32[np.where(Zvec <= cutoff)]
G33_ghost_atoms_z_l = G33[np.where(Zvec <= cutoff)]



# Combine ghost atoms with original vectors
Xvec = np.concatenate((X_ghost_atoms_x_l, X_ghost_atoms_y_l, X_ghost_atoms_z_l, Xvec, X_ghost_atoms_x_r, X_ghost_atoms_y_r, X_ghost_atoms_z_r))
Yvec = np.concatenate((Y_ghost_atoms_x_l, Y_ghost_atoms_y_l, Y_ghost_atoms_z_l, Yvec, Y_ghost_atoms_x_r, Y_ghost_atoms_y_r, Y_ghost_atoms_z_r))
Zvec = np.concatenate((Z_ghost_atoms_x_l, Z_ghost_atoms_y_l, Z_ghost_atoms_z_l, Zvec, Z_ghost_atoms_x_r, Z_ghost_atoms_y_r, Z_ghost_atoms_z_r))
G11 = np.concatenate((G11_ghost_atoms_x_l, G11_ghost_atoms_y_l, G11_ghost_atoms_z_l, G11, G11_ghost_atoms_x_r, G11_ghost_atoms_y_r, G11_ghost_atoms_z_r))
G12 = np.concatenate((G12_ghost_atoms_x_l, G12_ghost_atoms_y_l, G12_ghost_atoms_z_l, G12, G12_ghost_atoms_x_r, G12_ghost_atoms_y_r, G12_ghost_atoms_z_r))
G13 = np.concatenate((G13_ghost_atoms_x_l, G13_ghost_atoms_y_l, G13_ghost_atoms_z_l, G13, G13_ghost_atoms_x_r, G13_ghost_atoms_y_r, G13_ghost_atoms_z_r))
G21 = np.concatenate((G21_ghost_atoms_x_l, G21_ghost_atoms_y_l, G21_ghost_atoms_z_l, G21, G21_ghost_atoms_x_r, G21_ghost_atoms_y_r, G21_ghost_atoms_z_r))
G22 = np.concatenate((G22_ghost_atoms_x_l, G22_ghost_atoms_y_l, G22_ghost_atoms_z_l, G22, G22_ghost_atoms_x_r, G22_ghost_atoms_y_r, G22_ghost_atoms_z_r))
G23 = np.concatenate((G23_ghost_atoms_x_l, G23_ghost_atoms_y_l, G23_ghost_atoms_z_l, G23, G23_ghost_atoms_x_r, G23_ghost_atoms_y_r, G23_ghost_atoms_z_r))
G31 = np.concatenate((G31_ghost_atoms_x_l, G31_ghost_atoms_y_l, G31_ghost_atoms_z_l, G31, G31_ghost_atoms_x_r, G31_ghost_atoms_y_r, G31_ghost_atoms_z_r))
G32 = np.concatenate((G32_ghost_atoms_x_l, G32_ghost_atoms_y_l, G32_ghost_atoms_z_l, G32, G32_ghost_atoms_x_r, G32_ghost_atoms_y_r, G32_ghost_atoms_z_r))
G33 = np.concatenate((G33_ghost_atoms_x_l, G33_ghost_atoms_y_l, G33_ghost_atoms_z_l, G33, G33_ghost_atoms_x_r, G33_ghost_atoms_y_r, G33_ghost_atoms_z_r))




# Positions normalisation second times
Xvec = Xvec-min(Xvec) 
Yvec = Yvec-min(Yvec) 
Zvec = Zvec-min(Zvec)


# Calculating minimum and maximum X, Y and Z values
minX = float(min(Xvec))
minY = float(min(Yvec))
minZ = float(min(Zvec))


maxX = float(max(Xvec))
maxY = float(max(Yvec))
maxZ = float(max(Zvec))

#calculating minimum and maximum X, Y and Z values for ghost atoms
minX_ghost = float(min(X_ghost_atoms_x_l))
minY_ghost = float(min(Y_ghost_atoms_y_l))
minZ_ghost = float(min(Z_ghost_atoms_z_l))

maxX_ghost = float(max(X_ghost_atoms_x_l))
maxY_ghost = float(max(Y_ghost_atoms_y_l))
maxZ_ghost = float(max(Z_ghost_atoms_z_l))



# Calculating number of points in X, Y and Z direction for FFT
NXvalue = np.ceil((maxX - minX) / pixel_size) + 1
NYvalue = np.ceil((maxY - minY) / pixel_size) + 1
NZvalue = np.ceil((maxZ - minZ) / pixel_size) + 1

NX = int(NXvalue)
NY = int(NYvalue)
NZ = int(NZvalue)


# Calculating number of points of ghost atoms in X, Y and Z direction for FFT
NXvalue_ghost = np.ceil((maxX_ghost - minX_ghost) / pixel_size) 
NYvalue_ghost = np.ceil((maxY_ghost - minY_ghost) / pixel_size) 
NZvalue_ghost = np.ceil((maxZ_ghost - minZ_ghost) / pixel_size) 



NX_ghost = int(NXvalue_ghost) 
NY_ghost = int(NYvalue_ghost) 
NZ_ghost = int(NZvalue_ghost) 


# Generating grid points
x = np.linspace(minX, maxX, NX)
y = np.linspace(minY, maxY, NY)
z = np.linspace(minZ, maxZ, NZ)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

print (NX, NY, NZ)


# Définition des fonctions d'interpolation
def interpolate_grid(G):
    print(f"Interpolating {G} on process")  # Affichage pour suivre les tâches
    return griddata((Xvec, Yvec, Zvec), G, (X, Y, Z), method='linear')

# Liste des matrices à interpoler
G_components = [G11, G12, G13, G21, G22, G23, G31, G32, G33]

# Exécution parallèle
results = Parallel(n_jobs=-1)(delayed(interpolate_grid)(G) for G in G_components)

# Stockage des résultats
G11q, G12q, G13q, G21q, G22q, G23q, G31q, G32q, G33q = results


print('End of interpolation')



@nb.jit
def calculate_alpha_tensor(G11q, G12q, G13q, G21q, G22q, G23q, G31q, G32q, G33q, NX, NY, NZ, pixel_size):
    Alpha11 = np.zeros((NX, NY, NZ))
    Alpha12 = np.zeros((NX, NY, NZ))
    Alpha13 = np.zeros((NX, NY, NZ))
    Alpha21 = np.zeros((NX, NY, NZ))
    Alpha22 = np.zeros((NX, NY, NZ))
    Alpha23 = np.zeros((NX, NY, NZ))
    Alpha31 = np.zeros((NX, NY, NZ))
    Alpha32 = np.zeros((NX, NY, NZ))
    Alpha33 = np.zeros((NX, NY, NZ))

    for i in range(0, NX - 1, 1):
        for j in range(0, NY - 1, 1):
            for k in range(0, NZ - 1, 1):
                # Handling boundary conditions for i
                if (i == 0):
                    im = NX - 2
                    ip =  1
                elif (i == NX - 1):
                    im = NX - 2
                    ip = 1
                else:
                    im = i - 1
                    ip = i + 1
                # Handling boundary conditions for j
                if (j == 0):
                    jm = NY - 2
                    jp = 1
                elif (j == NY - 1):
                    jm = NY - 2
                    jp = 1
                else:
                    jm = j - 1
                    jp = j + 1
                # Handling boundary conditions for k 
                if (k == 0):  
                    km = NZ - 2
                    kp = 1
                elif (k == NZ - 1):
                    km = NZ - 2
                    kp = 1 
                else:
                    km = k - 1
                    kp = k + 1                    
                 

                Alpha11[i][j][k] = 0.5 * (-(G31q[i][jp][k] - G31q[i][jm][k]) + (G21q[i][j][kp] - G21q[i][j][km])) / pixel_size
                Alpha12[i][j][k] = 0.5 * ((G31q[ip][j][k] - G31q[im][j][k]) - (G11q[i][j][kp] - G11q[i][j][km])) / pixel_size
                Alpha13[i][j][k] = 0.5 * (-(G21q[ip][j][k] - G21q[im][j][k]) + (G11q[i][jp][k] - G11q[i][jm][k])) / pixel_size

                Alpha21[i][j][k] = 0.5 * (-(G32q[i][jp][k] - G32q[i][jm][k]) + (G22q[i][j][kp] - G22q[i][j][km])) / pixel_size
                Alpha22[i][j][k] = 0.5 * ((G32q[ip][j][k] - G32q[im][j][k]) - (G12q[i][j][kp] - G12q[i][j][km])) / pixel_size
                Alpha23[i][j][k] = 0.5 * (-(G22q[ip][j][k] - G22q[im][j][k]) + (G12q[i][jp][k] - G12q[i][jm][k])) / pixel_size

                Alpha31[i][j][k] = 0.5 * (-(G33q[i][jp][k] - G33q[i][jm][k]) + (G23q[i][j][kp] - G23q[i][j][km])) / pixel_size
                Alpha32[i][j][k] = 0.5 * ((G33q[ip][j][k] - G33q[im][j][k]) - (G13q[i][j][kp] - G13q[i][j][km])) / pixel_size
                Alpha33[i][j][k] = 0.5 * (-(G23q[ip][j][k] - G23q[im][j][k]) + (G13q[i][jp][k] - G13q[i][jm][k])) / pixel_size

    return Alpha11, Alpha12, Alpha13, Alpha21, Alpha22, Alpha23, Alpha31, Alpha32, Alpha33


# Compute Alpha tensors

A11, A12, A13, A21, A22, A23, A31, A32, A33 = calculate_alpha_tensor(G11q, G12q, G13q, G21q, G22q, G23q, G31q, G32q, G33q, NX, NY, NZ, pixel_size)


@nb.jit
def replace_nan_with_zero(NX, NY, NZ, array):
    
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if np.isnan(array[i, j, k]):
                    array[i, j, k] = 0


# Apply the JIT-compiled function to each array
replace_nan_with_zero(NX, NY, NZ, A11)
replace_nan_with_zero(NX, NY, NZ, A12)
replace_nan_with_zero(NX, NY, NZ, A13)
replace_nan_with_zero(NX, NY, NZ, A21)
replace_nan_with_zero(NX, NY, NZ, A22)
replace_nan_with_zero(NX, NY, NZ, A23)
replace_nan_with_zero(NX, NY, NZ, A31)
replace_nan_with_zero(NX, NY, NZ, A32)
replace_nan_with_zero(NX, NY, NZ, A33)


print ('*********************')
print ('End of interpolation')
print ('*********************')


X = X[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
Y = Y[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
Z = Z[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]

A11 = A11[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
A12 = A12[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
A13 = A13[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
A21 = A21[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
A22 = A22[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)] 
A23 = A23[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)] 
A31 = A31[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)] 
A32 = A32[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]
A33 = A33[int(NX_ghost) : int(NX-NX_ghost) , int(NY_ghost) : int(NY-NY_ghost), int(NZ_ghost) : int(NZ-NZ_ghost)]



#Reduice the dimension
X, Y, Z, A11, A12, A13, A21, A22, A23, A31, A32, A33 = X.flatten(), Y.flatten(), Z.flatten(), A11.flatten(), A12.flatten(), A13.flatten(), A21.flatten(), A22.flatten(), A23.flatten(), A31.flatten(), A32.flatten(), A33.flatten()                
      
#Normalize
X = X - min(X)
Y = Y - min(Y)


#Select the region containing a dislocation and extract the positions and components of the Nye tensor


a =  max(X)/2             #Boundaries of mask
b =  max(X)  
c = max(Y)/2 
d = max(Y)


#Mask for select the region of one dislocation
mask = (a <= X) & (X <= b) & (c <= Y) & (Y <= d) 


X = X[mask]  
Y = Y[mask]
Z = Z[mask]

A11 = A11[mask]
A12 = A12[mask]
A13 = A13[mask]
A21 = A21[mask]
A22 = A22[mask]
A23 = A23[mask]
A31 = A31[mask]
A32 = A32[mask]
A33 = A33[mask]

#Normalize
X = X - min(X)
Y = Y - min(Y)




# Empiler les tableaux en colonnes
data = np.column_stack((X, Y, Z, A11, A12, A13, A21, A22, A23, A31, A32, A33))

# Enregistrer dans un fichier texte
np.savetxt("Data_for_genereted_GB_images.txt", data, fmt="%.10f", delimiter='\t', header='X\tY\tZ\tA11\tA12\tA13\tA21\tA22\tA23\tA31\tA32\tA33', comments='')

print("Fichier data.txt enregistré avec succès.")



end_time = datetime.now()  # End timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")



