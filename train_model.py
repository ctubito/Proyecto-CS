import pandas as pd

#Cargar datos csv
data = pd.read_csv("data/penguins.csv")

#Visualizar datos
print(data.head())

#describir datos
print(data.describe())

#Preprocesar datos
data = data.dropna()
