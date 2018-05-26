import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from sklearn import preprocessing, cross_validation, neighbors
import _pickle as cPickle
import os
from sklearn.svm import LinearSVC
from keras.models import load_model


def import_check():
    print()
    print()
    print("Training Utils Imported")

def import_model(model_path):
    model = load_model(model_path)
    return model

model_path = 'C:/Users/Tanmay Patil/Downloads/Face Recognition/model/keras/model/facenet_keras.h5'
model = import_model(model_path)


def get_encodings(face):
    return calculate_encodings([face])

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def calculate_encodings(aligned_images, batch_size=1):
    
    aligned_images = prewhiten(np.array(aligned_images))
            
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    encodings = l2_normalize(np.concatenate(pd))

    return encodings

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train(face_encodings,labels,classifier = "KNN"):
    
    if(classifier == "SVC"):
        clf = trainLinearSVC(face_encodings,labels)
    
    elif(classifier == "DNN"):
        clf = trainDNN(face_encodings,labels)
    
    else:
        clf = trainKNearestNeighbors(face_encodings,labels)

    return clf

def trainKNearestNeighbors(face_encodings,labels,k=15):
    face_encodings,labels = unison_shuffled_copies(face_encodings,labels)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(face_encodings, labels,test_size=0)
    clf = neighbors.KNeighborsClassifier(n_neighbors = k,weights='distance')
    clf.fit(X_train, y_train)
    return clf

def trainLinearSVC(face_encodings,labels):
    face_encodings,labels = unison_shuffled_copies(face_encodings,labels)
    clf = LinearSVC(random_state=0)
    clf.fit(face_encodings,labels)
    return clf

def trainDNN(face_encodings,labels):
    return clf

def get_name_index(names,m):
    for n,name in enumerate(names):
        # print(name,names,m,n)
        if m == name:
            return n
        



