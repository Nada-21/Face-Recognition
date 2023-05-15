from PIL import Image
import glob
import numpy as np
from numpy import linalg as la
from pylab import *
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve 

#..........................................................................................................................................
train_images = []
test_images = []

for i in range(0,105):
    filename = r'C:\Users\power\Desktop\cv-task5\AllData\TrainingImages\Face' +str(i) + '.jpg'
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    train_images.append(im)

for filename in glob.glob(r'C:\Users\power\Desktop\cv-task5\AllData\TestingImages\*.jpg'):
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
    c = 0
    name =""
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
    nearest_distance = dist[nearest_face]
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
        elif index in range(75,105):
            name="Unknown"
    else:
        index = -1
        name = 'Unknown'

    return name, nearest_distance

#..................................................................................................................................
def testing_distance(k, zero_mean_test):
    matrixU = np.zeros((16384, k))
    c = 0
    Mean, Zero_mean_matrix, u_list = eigen()
    for val in range(k-1, -1, -1):
        matrixU[:, c] = u_list[val].flatten()
        c = c + 1
    w = np.dot(np.transpose(matrixU), np.transpose(zero_mean_test))
    weights = Reconstruct(k)
    original_w_k = weights
    dist = []
    for wt_vectors in original_w_k:
        dist.append(np.linalg.norm(wt_vectors-w.T))
    nearest_face = np.argmin(dist)
    nearest_distance = dist[nearest_face]
    return  nearest_distance


def create_roc_curve():
    distances = []
    labels = []
    Mean, Zero_mean_matrix, u_list = eigen()
    for i in range(TrainImages_num):
        test_image = train_images[i]
        zero_mean_test = test_image.flatten() - Mean
        nearest_distance = testing_distance(103, zero_mean_test)
        distances.append(nearest_distance)
        if i < 15:
            labels.append(0)  # Nada
        elif i < 30:
            labels.append(1)  # Kareman
        elif i < 45:
            labels.append(2)  # Naira
        elif i < 60:
            labels.append(3)  # Mayar
        elif i < 75:
            labels.append(4)  # Ghofran
        else:
            labels.append(5)  # Unknown
    fpr, tpr, thresholds = roc_curve(labels, distances)
    fig = plt.figure()
    plt.plot(fpr, tpr, linestyle='--',color='#d62728',)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve') 
    plt.show


create_roc_curve()