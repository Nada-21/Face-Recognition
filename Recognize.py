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

for i in range(0,90):
    filename = r'C:\Users\gufra\OneDrive\Desktop\3rd 2nd term\CV\cv-task5\AllData\TrainingImages\Face' +str(i) + '.jpg'
    im=Image.open(filename).convert('L')
    im= np.asarray(im,dtype=float)/255.0 
    train_images.append(im)

for i in range(0,60):
    filename =r'C:\Users\gufra\OneDrive\Desktop\3rd 2nd term\CV\cv-task5\AllData\TestingImages\Face' +str(i) + '.jpg'
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

    if np.min(dist) < threshold:  
        index = nearest_face
        if index in range(0,15):
            name="Nada"
            test_predict.append(0)
        elif index in range(15,30):
            name="kareman"
            test_predict.append(1)
        elif index in range(30,45):
            name="Naira"
            test_predict.append(2)
        elif index in range(45,60):
            name="Mayar"
            test_predict.append(3)
        elif index in range(60,75):
            name="Ghofran"
            test_predict.append(4)
        elif index in range(75,90):
            name="Unknown"
            test_predict.append(5)
    else:
        index = -1
        name = 'Unknown'
        test_predict.append(5)

    return name, test_predict

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
def ROC(labels,predicted):
    # Binarize the output
    labels = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5])
    predicted = label_binarize(predicted, classes=[0, 1, 2, 3, 4, 5])
    n_classes = labels.shape[1]
    target_names=['Nada','Kareman','Naira','Mayar','Ghofran','Unknown']
    fpr = {}
    tpr = {}
    roc_auc ={}
    lw =2 
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            labels[:, i],
            predicted[:, i],
            name=f"ROC curve for {target_names[i]}",
            color=color,ax=ax
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
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
    name, test_predict = Project(90, zero_mean_test, 80)  # threshold = 80

confusion(test_labels,test_predict)
ROC(test_labels,test_predict)