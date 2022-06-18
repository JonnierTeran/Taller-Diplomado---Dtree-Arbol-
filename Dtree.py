#Diplomado python 
#Autor: Jonnier Andres Teran Morales 
#ID No:502195
#Id:1003064599
#ID:correo:Jonnier.teran@upb.edu.co
#Cel:3255644212



import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus


carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
Datos = carseats.data
print(carseats.__doc__)

Datos['High_altas'] = np.where(Datos.Sales > 8, 0, 1)
Datos = Datos.drop(columns = 'Sales')

dfSheveloc= {'Malo':0, 'Medio':1, 'Bueno':2}
dfUrban= {'Si':1, 'No':0}
dfUs= {'Si':1, 'No':0}

Datos["ShelveLoc"] = Datos["ShelveLoc"].map(dfSheveloc)
Datos["Urban"] = Datos["Urban"].map(dfUrban)
Datos["US"] = Datos["US"].map(dfUs)


caracteristicas = ["CompPrice","Income","Advertising", "Population","Price","ShelveLoc","Age","Education","Urban","US"]

X = Datos[caracteristicas]
y = Datos["High_altas"]

xTrain = X[:320]
yTrain = y[:320]

xTest = X[320:]
yTest = y[320:]

dtree = DecisionTreeClassifier()
dtree  = dtree.fit(xTrain, yTrain)

prediction = dtree.predict([[136,70,12,171,152,1,44,18,1,1]])
miDato = tree.export_graphviz(dtree, out_file = None, feature_names= caracteristicas)
graph = pydotplus.graph_from_dot_data(miDato)
graph.write_png('mydecisiontree.png')
img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()