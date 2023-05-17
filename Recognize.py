from PIL import Image
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

#............................................. Load Traning and Testing Images .............................................................................................
train_images = []
test_images = []

for i in range(0,180):
    filename = r'C:\Users\power\Desktop\cv-task5\AllData\TrainingImages\Face' +str(i) + '.jpg'
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    train_images.append(im)

for i in range(0,60):
    filename =r'C:\Users\power\Desktop\cv-task5\AllData\TestingImages\Face' +str(i) + '.jpg'
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    test_images.append(im)

TrainImages_num = len(train_images)
TestImages_num = len(test_images)
#...........................................................................................................................................
test_labels = []
for i in range(10):
    test_labels.append(0)         #nada
for i in range(10):
    test_labels.append(1)         #kareman 
for i in range(10):
    test_labels.append(2)         #naira
for i in range(10):
    test_labels.append(3)         #mayar
for i in range(10):
    test_labels.append(4)         #ghofran 
for i in range(10):
    test_labels.append(5)         #unknown

#............................................... Getting Mean Face ang Eigen Faces .............................................................................................
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
    c = 0
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
test_predict = []
test_scores = []

def Project(k,zero_mean_test,threshold):
    matrixU = np.zeros((16384,k))
    c = 0
    name = ""
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

    # Step 1: Subtract the smallest distance from each distance in the group
    min_distance = min(dist)
    distances_shifted = [d - min_distance for d in dist]   
    # Step 2: Take the negative of each distance
    distances_neg = [-d for d in distances_shifted]
    # Step 3: Apply the softmax function to the negative distances
    probs = np.exp(distances_neg) / np.sum(np.exp(distances_neg))
    c1=np.mean(probs[0:30])              
    c2=np.mean(probs[30:60])
    c3=np.mean(probs[60:90])
    c4=np.mean(probs[90:120])
    c5=np.mean(probs[120:150])
    c6=np.mean(probs[150:])
    c1/=(c1+c2+c3+c4+c5+c6)
    c2/=(c1+c2+c3+c4+c5+c6)
    c3/=(c1+c2+c3+c4+c5+c6)
    c4/=(c1+c2+c3+c4+c5+c6)
    c5/=(c1+c2+c3+c4+c5+c6)
    c6/=(c1+c2+c3+c4+c5+c6)
    test_scores.append([c1,c2,c3,c4,c5,c6])

    nearest_face = np.argmin(dist)
    nearest_face_weights = original_w_k[nearest_face]
    zero_mean_test = zero_mean_test + np.transpose(Mean)
    face = np.dot(nearest_face_weights, np.transpose(matrixU))
    face = face + np.transpose(Mean)
    reshape_face = face.reshape(128, 128)
    
    if np.min(dist) < threshold:  
        index = nearest_face
        if index in range(0,30):
            name="Nada"
            test_predict.append(0)
        elif index in range(30,60):
            name="kareman"
            test_predict.append(1)
        elif index in range(60,90):
            name="Naira"
            test_predict.append(2)
        elif index in range(90,120):
            name="Mayar"
            test_predict.append(3)
        elif index in range(120,150):
            name="Ghofran" 
            test_predict.append(4)
        elif index in range(150,180):
            name="Unknown" 
            test_predict.append(5)
    else:
        index = -1
        name = 'Unknown'
        test_predict.append(5)


    return name, test_predict, test_scores

#.......................... Creating a confusion matrix, which compares the Testing_Labels and Testing_Predicted .........................
def confusion(labels,predicted):
    cm = confusion_matrix(labels, predicted)
    cm_df = pd.DataFrame(cm,
                         index = ['Nada','Kareman','Naira','Mayar','Ghofran','unknown'], 
                         columns = ['Nada','Kareman','Naira','Mayar','Ghofran','unknown'])
    fig = plt.figure(figsize=(6,6))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig("Confusion Matrix.jpg")

#.......................................... Creating ROC Curve For Multiclass ......................................................
def ROC(labels,scores):
    # Binarize the output
    y_test = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5])
    y_probs=np.array(scores)
    n_classes = y_test.shape[1]
    target_names=['Nada','Kareman','Naira','Mayar','Ghofran','Unknown']
    fpr = {}
    tpr = {}
    roc_auc ={}
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_test[:, i],
            y_probs[:, i],
            name=f"ROC curve for {target_names[i]}",color=color,ax=ax)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.savefig("ROC Curve.jpg")

#......................................................................................................................
Mean, Zero_mean_matrix, u_list = eigen()
for num in range(TestImages_num):
    t = test_images[num]
    test = t.flatten()
    zero_mean_test = test - np.transpose(Mean)
    name, test_predict, test_scores = Project(180, zero_mean_test, 80)  # threshold = 80

confusion(test_labels,test_predict)
ROC(test_labels,test_scores)