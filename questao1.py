import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# Preparação dos dados
dados = pd.read_csv('aerogerador.csv')

dados.head()

X = dados['X'].values
Y = dados['Y'].values

# Preparação para visualização 
plt.scatter(X,Y,label='Y(X)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


# Regressão em Grau 1
caracteristicas_1= PolynomialFeatures(degree=1)
X = X.reshape(-1, 1)
X_Polinomio_1 = caracteristicas_1.fit_transform(X)


modelo1 = LinearRegression()
modelo1.fit(X_Polinomio_1, Y)
Y_Polinomio_1 = modelo1.predict(X_Polinomio_1)

# Regressão em Grau 2
caracteristicas_2= PolynomialFeatures(degree=2)
X = X.reshape(-1, 1)
X_Polinomio_2 = caracteristicas_2.fit_transform(X)

modelo2 = LinearRegression()
modelo2.fit(X_Polinomio_2, Y)
Y_Polinomio_2 = modelo2.predict(X_Polinomio_2)


# Regressão em Grau 3
caracteristicas_3= PolynomialFeatures(degree=3)
X = X.reshape(-1, 1)
X_Polinomio_3 = caracteristicas_3.fit_transform(X)

modelo3 = LinearRegression()
modelo3.fit(X_Polinomio_3, Y)
Y_Polinomio_3 = modelo3.predict(X_Polinomio_3)

# Regressão em Grau 4
caracteristicas_4= PolynomialFeatures(degree=4)
X = X.reshape(-1, 1)
X_Polinomio_4 = caracteristicas_4.fit_transform(X)

modelo4 = LinearRegression()
modelo4.fit(X_Polinomio_4, Y)
Y_Polinomio_4 = modelo4.predict(X_Polinomio_4)

# Regressão em Grau 5
caracteristicas_5= PolynomialFeatures(degree=5)
X = X.reshape(-1, 1)
X_Polinomio_5 = caracteristicas_5.fit_transform(X)

modelo5 = LinearRegression()
modelo5.fit(X_Polinomio_5, Y)
Y_Polinomio_5 = modelo5.predict(X_Polinomio_5)

# Calculando R
r2_1 = r2_score(Y,Y_Polinomio_1)
r2_2 = r2_score(Y,Y_Polinomio_2)
r2_3 = r2_score(Y,Y_Polinomio_3)
r2_4 = r2_score(Y,Y_Polinomio_4)
r2_5 = r2_score(Y,Y_Polinomio_5)

# Calculando R adj.
r2Ajustado1 = 1 - (1-r2_1)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
r2Ajustado2 = 1 - (1-r2_2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
r2Ajustado3 = 1 - (1-r2_3)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
r2Ajustado4 = 1 - (1-r2_4)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
r2Ajustado5 = 1 - (1-r2_5)*(len(Y)-1)/(len(Y)-X.shape[1]-1)


#Plotar os Gráficos
fig, ax = plt.subplots(2, 3)
ax[0,0].scatter(X,Y)
ax[0,0].plot(X,Y_Polinomio_1,color='red',label='Ajuste')
ax[0,0].set_title('Grau 1 \n'+ 'r2: '+' %.16f' %r2_1 +'\n r2 Adj: ' +' %.16f' %r2Ajustado1)


ax[0,1].scatter(X,Y)
ax[0,1].plot(X,Y_Polinomio_2,color='red',label='Ajuste')
ax[0,1].set_title('Grau 2 \n'+ 'r2: '+' %.16f' %r2_2 +'\n r2 Adj: ' +' %.16f' %r2Ajustado2)


ax[0,2].scatter(X,Y)
ax[0,2].plot(X,Y_Polinomio_3,color='red',label='Ajuste')
ax[0,2].set_title('Grau 3 \n'+ 'r2: '+' %.16f' %r2_3 +'\n r2 Adj: ' +' %.16f' %r2Ajustado3)


ax[1,0].scatter(X,Y)
ax[1,0].plot(X,Y_Polinomio_4,color='red',label='Ajuste')
ax[1,0].set_title('Grau 4 \n'+ 'r2: '+' %.16f' %r2_4 +'\n r2 Adj: ' +' %.16f' %r2Ajustado4)


ax[1,1].scatter(X,Y)
ax[1,1].plot(X,Y_Polinomio_5,color='red',label='Ajuste')
ax[1,1].set_title('Grau 5 \n'+ 'r2: '+' %.16f' %r2_5 +'\n r2 Adj: ' +' %.16f' %r2Ajustado5)

# Saída dos Dados no Terminal

print("R2 (Grau 1) = ",r2_1)
print("R2 (Grau 2) = ",r2_2)
print("R2 (Grau 3) = ",r2_3)
print("R2 (Grau 4) = ",r2_4)
print("R2 (Grau 5) = ",r2_5)


print(r2Ajustado1)
print(r2Ajustado2)
print(r2Ajustado3)
print(r2Ajustado4)
print(r2Ajustado5)

plt.show()