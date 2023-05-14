from PIL import Image
import glob
import numpy as np
from numpy import linalg as la
from pylab import *
import matplotlib.pyplot as plt 

#..........................................................................................................................................
def griddisplay(image_list):
    rows = int(len(image_list) / 5) 
    fig1, axes_array = plt.subplots(rows, 5)
    fig1.set_size_inches(5,5)
    k=0
    for row in range(rows):
        for col in range(5):    
            im = np.array(Image.fromarray(image_list[k]).resize((100, 100), Image.ANTIALIAS))
            axes_array[row][col].imshow(im,cmap=plt.cm.gray) 
            axes_array[row][col].axis('off')
            k = k+1
    plt.show()
#..........................................................................................................................................
train_images = []
test_images = []

for i in range(0,75):
    filename = r'C:\cv-task5-1\AllData\TrainingImages\Face' +str(i) + '.jpg'
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    train_images.append(im)

for filename in glob.glob(r'C:\cv-task5-1\AllData\TestingImages\*.jpg'):
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    test_images.append(im)

TrainImages_num = len(train_images)
TestImages_num = len(test_images)
#...........................................................................................................................................
def eigen():
    column = 0
    u_list = []
    flattened_images  = []
    zero_mean = []

    for i in range(TrainImages_num):
        p=train_images[i].flatten()
        flattened_images.append(p)

    A_transpose = np.matrix(flattened_images)
    A = np.transpose(A_transpose)

    Mean= np.mean(A,1)

    Zero_mean_matrix= np.ones((16384,TrainImages_num))
    for values in flattened_images:
        zm= A[:,column] - Mean         # zm = values - mean
        zm = np.squeeze(zm)
        Zero_mean_matrix[:,column] =zm
        zm_images = zm.reshape(128,128)
        zero_mean.append(zm)
        column = column +1

    d = (np.dot(np.transpose(Zero_mean_matrix),Zero_mean_matrix))/425
    w2, v2 = la.eigh(d)
    for ev in v2:
        ev_transpose = np.transpose(np.matrix(ev))
        u = np.dot(Zero_mean_matrix,ev_transpose)                        
        u = u / np.linalg.norm(u)
        u_i= u.reshape(128,128)
        u_list.append(u_i)

    return Mean, Zero_mean_matrix, u_list

#..................................................................................................................................
def Reconstruct(k):
    weights=np.zeros((TrainImages_num,k))
    matrixU = np.zeros((16384,k))
    c =0
    Mean, Zero_mean_matrix, u_list = eigen()
    for val in range(k-1,-1,-1):
        matrixU[:,c] = u_list[val].flatten()
        c = c+1
    rec_face = [] 
    for face_num in range(0,TrainImages_num):
        w = np.dot(np.transpose(matrixU) ,Zero_mean_matrix[:,face_num])
        weights[face_num,:] =w
        face = np.dot(w, np.transpose(matrixU))
        minf = np.min(face)
        maxf = np.max(face)
        face = face-float(minf)
        face = face/float((maxf-minf))
        face = face + np.transpose(Mean)
        reshape_face = face.reshape(128,128)
        rec_face.append( reshape_face)

    return weights

#......................................................................................................................................
def Project(k,zero_mean_test,threshold):
    matrixU = np.zeros((16384,k))
    c =0
    name=""
    Mean, Zero_mean_matrix, u_list = eigen()
    for val in range(k-1,-1,-1):
        matrixU[:,c] = u_list[val].flatten()
        c = c+1
    w = np.dot(np.transpose(matrixU) ,np.transpose(zero_mean_test))
    weights = Reconstruct(k)
    original_w_k = weights
    dist = []
    
    for wt_vectors in original_w_k:
        dist.append(np.linalg.norm(wt_vectors - w.T))

    nearest_face = np.argmin(dist)
    nearest_face_weights = original_w_k[nearest_face]

    zero_mean_test =zero_mean_test + np.transpose(Mean)
    zero_mean_test = zero_mean_test.reshape(128,128)

    face = np.dot(nearest_face_weights, np.transpose(matrixU))
    face = face + np.transpose(Mean)
    reshape_face = face.reshape(128,128)

    if np.min(dist) < threshold:  # Nonface
        index = nearest_face
        if index in range(0,15):
            name="Nada"
        elif index in range(15,30):
            name="kareman"
        elif index in range(30,45):
            name="Naira"
        elif index in range(45,60):
            name="Mayar"
        elif index in range(60,75):
            name="Ghofran"
        else:
            index = -1
            name = 'Unknown'

    return name
    

def runs(k):
    for num in range(0,TestImages_num): # t in test_images:
        Mean, Zero_mean_matrix, u_list = eigen()
        t = test_images[num]
        test = t.flatten()
        zero_mean_test = test-np.transpose(Mean)
        name = Project(k,zero_mean_test,80)  #threshold =80

# runs(39)

