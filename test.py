#! C:\Users\Asus-Pc\OneDrive - Universidad Politécnica de Madrid\Escritorio\python\myenv\Scripts\python.exe

import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt

data_set_train=np.array(pd.read_csv('ItalyPowerDemand/ItalyPowerDemand_TRAIN.tsv',sep='\t'))
data_set_test=np.array(pd.read_csv('ItalyPowerDemand/ItalyPowerDemand_TEST.tsv',sep='\t'))
     # get the training data
train_data=data_set_train[:,1:]
train_classes=data_set_train[:,0]

def continuous():
    y=[i for i in range(0,len(train_data[0,:]))]
    for j in range(0,len(train_data[:])):
        if train_classes[j] == 1.0:
            plt.plot(y,(train_data[j,:]), color='black')
            c='black'
        else:
            plt.plot(y,(train_data[j,:]), color='green')
            c='green'
        for i in range(len(y)):
            y[i]=y[i]+len(y)
        try:
            plt.plot([y[0]-1,y[0]],[(train_data[j,23]),(train_data[j+1,0])], color=c)
        except IndexError:
            break
    plt.show()


def mean():
    mean_data_1=np.zeros([24])
    mean_data_2=np.zeros([24])
    n1=0
    n2=0
    for j in  range(0,len(train_data[:])):
        for i in range(0,len(train_data[0,:])):
            if train_classes[j] == 1:
                mean_data_1[i]=mean_data_1[i]+train_data[j,i]
                n1=n1+1/24
            else:
                mean_data_2[i]=mean_data_2[i]+train_data[j,i]
                n2=n2+1/24
    mean_data_1=mean_data_1/n1
    mean_data_2=mean_data_2/n2

    sd_data_1=np.zeros([24])
    sd_data_2=np.zeros([24])
    for j in  range(0,len(train_data[:])):
        for i in range(0,len(train_data[0,:])):
            if train_classes[j] == 1:
                sd_data_1[i]=sd_data_1[i]+(train_data[j,i]-mean_data_1[i])**2
            else:
                sd_data_2[i]=sd_data_2[i]+(train_data[j,i]-mean_data_2[i])**2

    sd_data_1=np.sqrt(sd_data_1/n1)
    sd_data_2=np.sqrt(sd_data_2/n2)
    print(sd_data_1)
    print(sd_data_2)

    y= [i for i in range(0,24)]
    plt.errorbar(y,mean_data_1,yerr=sd_data_1,color='black',ecolor='black')
    plt.errorbar(y,mean_data_2,yerr=sd_data_2,color='green',ecolor='green')

    plt.show()

continuous()
#for i in range(0,len(train_data)):
#    if train_classes[i] == 1:
#        mp.plot(train_data[i,:],color='black')
#    else:
#        mp.plot(train_data[i,:],color='green')
#mp.show()

#TODOS: 
#Get a baseline, to compare against
#Retrieve from the samples the seasonality, randomness and trend if there is any
#Try against the training dataset
#With the test dataset apply the trend, seasonality and randomness 
