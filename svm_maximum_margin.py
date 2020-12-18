# Algunos ejemplos de SVM en este enlace: https://scikit-learn.org/stable/auto_examples/

# En este ejemplo mostraremos el hiperplano que separa con el mayor margen posible a un dataset
# de dos clases usando el clasificador SVM (Support Vector Machine) con un kernel lineal

# Se importan las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Método principal de la aplicación
if __name__ == '__main__':
    # Se crean 40 puntos aleatorios
    # La variable X contiene los 40 puntos (coordenada x,y)
    # La variable y contiene las clases (0 o 1) -> centers=2
    # Siempre se genera el mismo conjunto aleatorio (random_state)
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    print('Dataset: ', X)
    print('\nClases: ', y)

    # Entrenamos el modelo con un kernel lineal
    # C es un parámetro de regularización
    # En realidad no regularizamos, porque es un ejemplo sencillo
    clasificador = svm.SVC(kernel='linear', C=1000)
    clasificador.fit(X, y)

    # Se crea un gráfico de tipo "scatter plot"
    # Más información: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
    # s indica el tamaño de los puntos en la gráfica
    # colormap indica el esquema de color que se usará para cada clase de punto
    colormap = np.array(['r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], c=colormap[y], s=30)

    # Se pinta la función de decisión
    # plt.gca() obtiene los ejes del gráfico
    ax = plt.gca()

    # Se obtienen los límites de los ejes x, y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Se crea un grid para evaluar el modelo
    # La función linspace devuelve números distribuidos por igual y espaciados en un intervalo especificado
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    # La función meshgrid devuelve una matriz de coordenadas a partir de vectores de coordenadas
    YY, XX = np.meshgrid(yy, xx)

    # Se añade al plot la malla (mesh) de puntos generados (tamaño de punto 2)
    plt.plot(XX, YY, marker='.', color='k', markersize=2, linestyle='none')

    # La función vstack hace un stack de los arrays en una secuencia vertical (row wise)
    # La función ravel devuelve un array plano contiguo (1 dimensión) con todos los elementos de entrada
    # La función T calcula la transpuesta
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # La función decision_function predice las puntuaciones de confianza de las muestras
    # La puntuación de la confianza es la distancia (con signo) de la muestra al hiperplano
    Z = clasificador.decision_function(xy).reshape(XX.shape)

    # Se añade al gráfico la frontera de decisión del SVM y los márgenes (inferior y superior)
    ax.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'])

    # Se añaden los vectores de soporte
    ax.scatter(clasificador.support_vectors_[:, 0], clasificador.support_vectors_[:, 1], s=100,
               linewidth=2.5, facecolors='none', edgecolors='k')

    # Se muestra el gráfico del SVM
    plt.show()
