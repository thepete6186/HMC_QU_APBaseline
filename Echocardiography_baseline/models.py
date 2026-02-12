# -*- coding: utf-8 -*-
"""
Recreation for baseline use. 
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D 
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

Scoring = {'AUC':'roc_auc', 'Accuracy':'accuracy', 'Recall': 'recall', 'F1-Score': 'f1', 'Precision': 'precision'}

def Resnet34():

      model = Sequential(learning_rate = 1e-3)

      model.add(Conv2D(filters = 64, kernel_size = 7, strides = 2, padding ='same',activation= 'relu', input_shape = (224, 224, 3)))
      model.add(MaxPooling2D(pool_size = 3, strides = 2, padding ='same'))
      model.add(Residualblock(64, 2))
      model.add(Residualblock(64))
      model.add(Residualblock(64))
      model.add(Residualblock(128, 2))
      model.add(Residualblock(128))
      model.add(Residualblock(128))
      model.add(Residualblock(128))
      model.add(Residualblock(256, 2))
      model.add(Residualblock(256))
      model.add(Residualblock(256))
      model.add(Residualblock(256))
      model.add(Residualblock(256))
      model.add(Residualblock(256))
      model.add(Residualblock(512, 2))
      model.add(Residualblock(512))
      model.add(Residualblock(512))
      model.add(GlobalAveragePooling2D())
      model.add(Dense(2, activation = 'softmax'))

      model.summary()
      opti = optimizers.Momentum(learning_rate = 1e-3, momentum = 0.9)
      model.compile(loss = 'categorical_crossentropy', optimizer = opti, metrics = ['accuracy'])

      return model

class Residualblock(tf.keras.layers.Layer):
    def __init__(self, filter_size, stride=1):
        super(Residualblock, self).__init__()
        fpadding = 'same'
        if stride != 1:
            fpadding = 'valid'
        self.conv_sequence = Sequential([
            Conv2D(filters=filter_size, kernel_size=3, strides=stride, padding=fpadding),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, input):
        x = self.conv_sequence(input)
        if input.shape[-1] == x.shape[-1]:
         x += input
        return x

def Resnet34_train(X_train, y_train, REFIT):
    
    def resnet34_model():
       return Resnet34()
    
    model = KerasClassifier(model = resnet34_model, verbose = 0) 
    param_grid = dict(epochs = [25, 50, 75, 100])
    grid_search = GridSearchCV(estimator = model, n_jobs = 1, param_grid = param_grid, scoring = Scoring, refit = REFIT, cv = 5)
    grid_search = grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_
   

def cnn1D_model(inpt_dim, outpt_dim, kernel_size = 5, filter_size = 8, learning_rate = 1e-1):
          
    model = Sequential()
    
    model.add(Convolution1D(filters = filter_size, kernel_size = kernel_size, padding ='same',activation= 'relu', input_shape = (inpt_dim, 1)))
    model.add(MaxPooling1D(pool_size = 2, strides = 2))
    model.add(Convolution1D(filters = int(filter_size/2), kernel_size = kernel_size, padding ='same',activation= 'relu'))
    model.add(MaxPooling1D(pool_size = 2, strides = 2))
    model.add(Flatten(name='flat'))
    flat_layer = model.output_shape[1]
    print(flat_layer) 
    xx = int(flat_layer - ((flat_layer - outpt_dim) / 2))
    model.add(Dense(xx, activation = 'relu'))
    model.add(Dense(outpt_dim-1, activation = 'softmax'))

    model.summary()
            
    opti = optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer = opti, metrics = ['accuracy'])
    
    return model


def CNN_train(X_train, y_train, REFIT):
    
    print('initiating function')
    
    def cnn_model (kernel_size = 5, filter_size = 8, learning_rate = 1e-1):
       return cnn1D_model(
            inpt_dim = X_train.shape[1], outpt_dim = len(np.unique(y_train)),
        kernel_size = kernel_size, filter_size = filter_size, learning_rate = learning_rate

       )
    
    
    print('...Training... CNN')    
    model = KerasClassifier(model = cnn_model, verbose = 0) 
    print('model finished loading')
    kernel_size = [3, 5, 7, 9, 11, 13, 15]
    filter_size = [4, 8, 12, 16, 24, 32]
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    print('no problem yet')
    param_grid = dict(model__kernel_size = kernel_size, model__filter_size = filter_size, model__learning_rate = learning_rate, epochs = [25, 50, 75, 100])
    grid_search = GridSearchCV(estimator = model, n_jobs = 1, param_grid = param_grid, scoring = Scoring, refit = REFIT, cv = 5)
    print('grid search loaded')
    grid_search = grid_search.fit(X_train, y_train)
    print('complete')

    
    return grid_search.best_estimator_, grid_search.best_params_ 


def SVM_train(X_train, y_train, REFIT):
  params_grid = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 'C': [1, 10, 100, 1000]}]                 
  svm_model = GridSearchCV(SVC(), params_grid, n_jobs = 1, scoring = Scoring, refit = REFIT, cv = 5)
  print('SVM Train')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished')
  
  return svm_model.best_params_, svm_model.best_estimator_


def DT_train(X_train, y_train, REFIT):
  params_grid = [{'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'], 
                  'max_features': ['auto', 'sqrt', 'log2']}]                 
  
  dt_model = GridSearchCV(DecisionTreeClassifier(), params_grid, n_jobs = 1, scoring = Scoring, refit = REFIT, cv = 5)
  print('DT Train')
  dt_model.fit(X_train, y_train)
  print('DT Train Finished')
  
  return dt_model.best_params_, dt_model.best_estimator_


def RF_train(X_train, y_train, REFIT):
  params_grid = [{'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],'criterion': ['gini', 'entropy'], 
                  'max_features': ['sqrt', 'log2', None], 'class_weight': ['balanced', 'balanced_subsample'],
                  'warm_start': [False, True], 'bootstrap': [False, True]}]                 
  
  RF_model = GridSearchCV(RandomForestClassifier(), params_grid, n_jobs = 1, scoring = Scoring, refit = REFIT, cv = 5)
  print('RF Train')
  RF_model.fit(X_train, y_train)
  print('RF Train Finished')
  
  return RF_model.best_params_, RF_model.best_estimator_


def KNN_train(X_train, y_train, REFIT):
  params_grid = [{'algorithm': ['ball_tree', 'kd_tree', 'brute'],'n_neighbors': [5, 10, 15, 20, 25, 30], 
                  'weights': ['uniform', 'distance'], 'p':[1, 2]}]                 
  
  knn_model = GridSearchCV(KNeighborsClassifier(), params_grid, n_jobs = 1, scoring = Scoring, refit = REFIT, cv = 5)
  print('KNN Train')
  knn_model.fit(X_train, y_train)
  print('KNN Train Finished')
  
  return knn_model.best_params_, knn_model.best_estimator_


def performance_metrics(CM):
    CM = CM.astype('int64')
    TN = CM[0,0]
    FP = CM[0,1]
    FN = CM[1,0]
    TP = CM[1,1]
    
    Sensitivity = (TP / (TP + FN))
    Specificity = (TN / (TN + FP)) 
    Precision = (TP / (TP + FP))
    F1 = (2*TP) / (2*TP + FP +FN)
    beta = 2
    F2 = (1+beta**2) * ((Precision * Sensitivity) / (beta**2 * Precision + Sensitivity))
    ACC = (TP + TN) / (TP + TN + FN +FP)
            
    metrics = [Sensitivity*100, Specificity*100, Precision*100, F1*100, F2*100, ACC*100]
    
    return metrics

import warnings
warnings.filterwarnings("ignore")