import os
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
import cv2 as cv
import numpy as np
from skimage.feature import hog
import colorsys
from sklearn import model_selection, metrics, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

########################## Import image dataset from a folder (each sub folder is a classe)
def lire(path_in):
    dossier =  path_in
    dir_dossier = os.listdir(dossier)
    
    base = []

    for i in dir_dossier:
        sous_dossier = os.path.join(dossier,i)
        try:
            image = cv.imread(sous_dossier)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            base.append(image)
        except:
            print(" Error !")
    return base

########################## Pre-processing function of image dataset from folder (each sub folder is a classe)

def process(path_in, size) :
    dossier =  path_in
    
    classe = []
    vect_cara = []
    imghog = []
    img = []

    for i in os.listdir(path_in) :
        sous_dossier = os.path.join(dossier,i)
        
        for j in os.listdir(sous_dossier) :
            sous_sous_dossier = os.path.join(sous_dossier,j)
            
            try:
                image = cv.imread(sous_sous_dossier)
                image = cv.resize(image, size)

                img_filtre_median = cv.medianBlur(image, 3)
                img_ycbcr = cv.cvtColor(img_filtre_median, cv.COLOR_RGB2YCrCb)

                fv, img_hog = hog(img_ycbcr/255, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (1, 1), visualize = True, multichannel = True)
                img_hog = np.array(img_hog)
                
                vect_cara.append(fv)
                #classe.append(str(i)) # Les classes en caractères ou bien en nombre.
                if i == 'classe avertissement de danger' : ## Change your classe names
                    classe.append(1)
                elif i == 'classe interdicition ou restriction' :
                    classe.append(2)
                elif i == 'classe obligation' :
                    classe.append(3)
                
                imghog.append([img_hog],)
                img.append([image],)

            except :
                print("Erreur !")
        
    return [classe, vect_cara, imghog, img]

## Function call
[y, x, imghog, img] = process(r'C:/Users/surface book 2/Desktop/Traffic routier', (64, 64))

########################## Data subdivision
x_train, x_test, y_train, y_test = model_selection.train_test_split(np.array(x), np.array(y), train_size = 2/3, random_state = 1956)

########################## Save the processed image

path_out = r'C:/Users/surface book 2/Desktop/'

### Ordered following the classe
def sauvegarder(element, path_out, directory) : # element is the img or imghog which is the image dataset
    path_dossier = os.path.join(path_out, directory)
    os.mkdir(path_dossier)

    for i in range(len(element)) :
        direction = str(i)
        path_sous_dossier = os.path.join(path_dossier, direction)
        if not os.path.exists(path_sous_dossier) :
            os.mkdir(path_sous_dossier)

        for j in range(len(element[i][:])) :
            cv.imwrite(path_sous_dossier + '\\' + f"{j:4d}.png", element[i][j])
    
    return element

### Unordered
def sauvegarder(element, path_out, directory) :
    path_dossier = os.path.join(path_out, directory)
    os.mkdir(path_dossier)

    for i in range(len(element)) :
        for j in range(len(element[i][:])) :
            cv.imwrite(path_dossier + '\\' + f"{j:4d}.png", element[i][j])
    
    return element

########################## Split dataset contained in a folder

parent_folder = r'C:/Users/surface book 2/Desktop/' # La direction du dossier parent
file_data = "Traffic_processed" # Le nom dossier base de donnée
file_split = "Split" # Le nom dossier de repartition
input_folder = os.path.join(parent_folder,file_data)
output_folder = os.path.join(parent_folder,file_split)
ratio = (.67, .33) # Pourcentage de repartition train, val, test
group_prefix = None
seed = 42

splitfolders.ratio(input_folder, output_folder, seed, ratio, group_prefix)