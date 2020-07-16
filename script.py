# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:58:46 2020

@author: Mahdi Tanbakuchi
"""

import numpy as np
import random 
import matplotlib.pyplot as plt
from itertools import permutations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle

random.seed(0)
Fr = np.linspace(0,7,30)
Fr = np.concatenate((Fr,np.linspace(11,15,30)))
Fw = np.linspace(8,10,30)
Fr = np.random.permutation(Fr)
Fw = np.random.permutation(Fw)
scale = np.linspace(1,100,100)
scale = np.random.permutation(scale)

fs = 32       # in hertz    
duration=8   # in seconds
ch = 8       # Number of channels

def datagen(freq,coef,fs=32,duration=8):    
    t = np.arange(0,duration,1/fs)
    y = np.zeros(t.shape)
    if(np.linalg.norm(coef)!=1):
        coefnorm = coef/np.linalg.norm(coef)
    for i,j in zip(freq,coefnorm):
        y+=j*np.cos(2*np.pi*i*t)
    return y
        

N = 5 
datasetsize = 10000
clr = 0.2
t = np.arange(0,duration,1/fs)
Frd = permutations(Fr,N)
Fwd = permutations(Fw,N)
scale1=permutations(scale,N)
scale = np.random.permutation(scale)
scale2=permutations(scale,N)
posdatasize = np.int32(datasetsize*(1-clr))
negdatasize = np.int32(datasetsize*clr)
posdata = []
negdata = []
for i in range(posdatasize):
    posdata.append(datagen(next(Frd),next(scale1)))
for i in range(negdatasize):
    negdata.append(datagen(next(Fwd),next(scale2)))

a1 = plt.subplot(121)
plt.plot(t,posdata[0])
a1.set_title("Normal Case")
a2 = plt.subplot(122)
plt.plot(t,negdata[0])
a2.set_title("Abnormal case")


weakpositer = permutations(posdata,ch) 
posdataweak = np.zeros((posdatasize,ch,*t.shape))
classid = np.zeros((posdatasize))
for i in range(posdatasize):
    posdataweak[i]=np.array(next(weakpositer))


nums = random.choices(range(1,4),k=negdatasize)
positions = [random.sample(range(0,8),k=i) for i in nums]
posdata2 = iter(np.random.permutation(posdata))
negdata2 = iter(np.random.permutation(negdata))
negdataweak=[]
negdata3=np.zeros((negdatasize,ch,*t.shape))
classid = np.concatenate((classid,np.ones((negdatasize))))
index = 0
w0 = 1/posdatasize
w1 = 1/negdatasize
for pos in positions:
    for i in range(8):
        if(i in pos):
            try:
                negdata3[index,i,:] = next(negdata2)
            except StopIteration:
                negdata2 = iter(np.random.permutation(negdata))
                negdata3[index,i,:] = next(negdata2)
        else:
            try:
                negdata3[index,i,:] = next(posdata2)
            except StopIteration:
                posdata2 = iter(np.random.permutation(posdata))
                negdata3[index,i,:] = next(posdata2)
    index+=1
            
negdataweak = negdata3
datasetweak = np.concatenate((posdataweak,negdataweak))
datasetweak = np.expand_dims(datasetweak,-1)


dummy = np.expand_dims(posdataweak,-1)
input_shape = dummy.shape[1:]

model = keras.Sequential(
    [
        # Feature extraction blocks
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(1,4),strides=(1,3)),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(1,4),strides=(1,3)),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.Conv2D(32, kernel_size=(1,3), activation="relu"),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(1,4),strides=(1,3)),
        # Classification Block
        layers.Conv2D(2, kernel_size=(1,3), activation="relu"),
        layers.AveragePooling2D(pool_size=(1,4),strides=(1,3)),
        layers.MaxPooling2D(pool_size=(ch,1)),
        layers.Flatten(),
        layers.Softmax()
    ]
)
model.summary()
model.save("Raw_Model")


batch_size = 300
epochs = 1500
num_classes = 2
model = keras.models.load_model("Raw_Model")
ylabel = keras.utils.to_categorical(classid, num_classes)
X_train,X_test,Y_train,Y_test = train_test_split(datasetweak,ylabel,test_size = 0.2 , random_state=2)
sgd = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True, name="SGD"
)
model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy","AUC"])
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

class_weight = {0: w0, 1: w1}
weaklabel = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[callback],class_weight=class_weight)

model.save("Unsorted_dataset")


# New Antecedent/Consequent objects hold universe variables and membership
# functions
freq = ctrl.Antecedent(np.arange(0, 16, 1/100), 'freq')
Power = ctrl.Antecedent(np.arange(0,100,1/10),"Power")
Siezure=ctrl.Consequent(np.arange(0, 2, 1/100), 'Siezure')

freq["Low"] = fuzz.gaussmf(freq.universe,0,5)
freq["Mid"] = fuzz.gaussmf(freq.universe,9,0.5)
freq["High"] = fuzz.gaussmf(freq.universe,15,2.5)

Power["lims"] = fuzz.trimf(Power.universe,[0,100,100])

Siezure["Low"] = fuzz.gaussmf(Siezure.universe,0,0.1)
Siezure["High"] = fuzz.gaussmf(Siezure.universe,1,0.1)

rule1 = ctrl.Rule(freq['Low'] | freq['High'] & Power['lims'],Siezure['Low'])
rule2 = ctrl.Rule(freq['Mid'] & Power['lims'], Siezure['High'])
Siezureness_ctrl = ctrl.ControlSystem([rule1,rule2])
Siezureness = ctrl.ControlSystemSimulation(Siezureness_ctrl)
Siezureness.inputs({'freq':10,'Power':10})
Siezureness.compute()
print(Siezureness.output['Siezure'])

freq.view()
ax = Siezure.view()
Power.view()

def sortdata(EEGdata,fs):
    siezuremeasure = np.zeros((EEGdata.shape[0]))
    sortedEEGdata = np.zeros(EEGdata.shape)
    for k in range(EEGdata.shape[0]):
        fftdata = np.fft.fft(EEGdata[k,:])
        for i in range(*fftdata.shape):
            Siezureness.inputs({'freq':i/16,'Power':np.abs(fftdata[i])})
            Siezureness.compute()
            siezuremeasure[k] += Siezureness.output['Siezure']
    sort = np.sort(siezuremeasure)[::-1]    
    for k in range(sort.shape[0]):
        h = sort[k]==siezuremeasure
        sortedEEGdata[k,:] = EEGdata[np.where(h)[0][0],:]
    return sortedEEGdata
def sortdataset(dataset,fs):
    sorteddataset = np.zeros(dataset.shape)
    for i in range(dataset.shape[0]):
        sorteddataset[i] = sortdata(dataset[i],fs)
    return sorteddataset

orderedposdataweak = sortdataset(posdataweak,fs)

with open("posdatasetsorted","wb") as file:
    pickle.dump(orderedposdataweak,file)

orederednegdataweak = sortdataset(negdataweak,fs)
with open("negdatasetsorted","wb") as file:
    pickle.dump(orederednegdataweak,file)
    
datasetweakordered = np.concatenate((orderedposdataweak,orederednegdataweak))
datasetweakordered = np.expand_dims(datasetweakordered,-1)
datasetweakordered.shape

model = keras.models.load_model("Raw_Model")
X_train,X_test,Y_train,Y_test = train_test_split(datasetweakordered, ylabel,test_size = 0.2 , random_state=2)
sgd = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True, name="SGD"
)
model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy","AUC"])
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

class_weight = {0: w0, 1: w1}
fuzzyordered = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[callback],class_weight=class_weight)

model.save("sorteddataset")


negdata3sorted=np.zeros((negdatasize,ch,*t.shape))
index = 0
w0 = 1/posdatasize
w1 = 1/negdatasize
for pos in positions:
    num = 0
    for p in pos:
        try:
            negdata3sorted[index,num,:] = next(negdata2)
            num+=1
        except:
            negdata2 = iter(np.random.permutation(negdata))
            negdata3sorted[index,num,:] = next(negdata2)
            num+=1
    try:
        negdata3sorted[index,num,:] = next(posdata2)
        num+=1
    except:
        posdata2 = iter(np.random.permutation(posdata))
        negdata3sorted[index,num,:] = next(posdata2)
        num+=1
    index+=1

sorteddataset= np.expand_dims(np.concatenate((posdataweak,negdata3sorted)),-1)
sorteddataset.shape


model = keras.models.load_model("Raw_Model")
X_train,X_test,Y_train,Y_test = train_test_split(sorteddataset, ylabel,test_size = 0.2 , random_state=2)
sgd = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True, name="SGD"
)
model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy","AUC"])
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

class_weight = {0: w0, 1: w1}
sortedmanual = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[callback],class_weight=class_weight)

model.save("sortedmanual")

with open("history","wb") as file:
    pickle.dumps([sortedmanual,fuzzyordered,weaklabel])