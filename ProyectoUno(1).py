import pandas as pd
#import numpy as np
from neuron import NN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def datasetConfig(archivo):
    datos = pd.read_csv(archivo, header=None)
    dataset = datos.values
    
    x = dataset[:,:-1]
    y = dataset[:,-1]
    
    x = x.astype(str)
    y = y.reshape((len(y),1))
    return x,y

def prepararEntradas(xTrain, xTest):
    oe = OrdinalEncoder()
    oe.fit(xTrain)
    xTrainEnc = oe.transform(xTrain)
    xTestEnc = oe.transform(xTest)
    return xTrainEnc, xTestEnc

def prepararSalidas(yTrain, yTest):
    le = LabelEncoder()
    le.fit(yTrain)
    yTrainEnc = le.transform(yTrain)
    yTestEnc = le.transform(yTest)
    return yTrainEnc, yTestEnc

x, y = datasetConfig("DataSets/tic-tac-toe.data")
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size= 0.33, random_state=1)

xTrainEnc, xTestEnc = prepararEntradas(xTrain, xTest)
yTrainEnc, yTestEnc = prepararSalidas(yTrain, yTest)

red = NN(9, 18, 1, 0.4, 0.8)

entrenamientos = 502
contador = 0

for j in range(entrenamientos):
    print(contador,"...")
    contador += 1
    for i in range(len(xTrainEnc)): #se pasaron 100 argumentos
        salida, act = red.calcNetOutput(list(xTrainEnc[i].ravel()),True)
        red.trainingEpisode(list(yTrainEnc[i].ravel()), salida, act, list(xTrainEnc[i].ravel()))

p = 0
n = 0

for i in range(len(xTestEnc)):
    salida = red.calcNetOutput(list(xTestEnc[i].ravel()))
    #print(list(xTestEnc[i].ravel()))
    #print("salida: ", salida)
    '''if(salida[0] > 0.08): #and salida[1] < 0.3):
        p += 1
    elif(salida[0] < 0.3):
        n += 1'''
    if(yTestEnc[i] == 0):
        if(salida[0] < 0.08):
            p += 1
    elif(yTestEnc[i] == 1):
        if(salida[0] > 0.3):
            n += 1

print("porcentaje red neuronal: ",(p+n)/317)
print("positivo: ",p / 114)
print("negativo: ",n / 203)
#print("total: ",(p+n))
#print("p: ", p)
#print("n: ", n)

#82.33 #500 #NN(9, 18, 1, 0.4, 0.8)
#79.0107 #500 #NN(9, 18, 1, 0.7, 0.8)
#75.3943 #500 #NN(9, 18, 1, 0.7, 0.7)
#74.7634 #500 #NN(9, 18, 1, 0.5, 0.8)
#75.7009 #500 #NN(9, 18, 1, 0.3, 0.8)
#77.2870 #500 #NN(9, 18, 1, 0.35,0.8)
#77.2870 #500 #NN(9, 18, 1, 0.42, 0.8)
#78.8643 #500 #NN(9, 18, 1, 0.385, 0.8)
#0.817034
#0.706624

'''uno = 0
cero = 0

for i in range(len(yTestEnc)):
    if(yTestEnc[i] == 1):
        uno += 1
    elif(yTestEnc[i] == 0):
        cero += 1
        
print(uno)
print(cero)'''