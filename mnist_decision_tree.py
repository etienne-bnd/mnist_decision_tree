from IPython.display import set_matplotlib_formats, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
# c'est un module pour apprendre le machine learning écrit par les gens qui ont écrit le livre au dessus
from cycler import cycler
# cycler c'est pour matplotlib pour pouvoir faire des affichages 
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# cette fonction servira pour couper le dataset en le dataset d'entraînement / test / validation



set_matplotlib_formats('pdf', 'png')
# on définit le format des images matplotlib sous forme de pdf ou de png
plt.rcParams['savefig.dpi'] = 300
# définit la résolution des figures
plt.rcParams['image.cmap'] = "viridis"
# on défini la colormap pour les images quelles couleurs vont être utilisées
plt.rcParams['image.interpolation'] = "none"
# pas d'interpolation
plt.rcParams['savefig.bbox'] = "tight"
# on ne coupe rien avec l'image englobande
plt.rcParams['lines.linewidth'] = 2
# défini l'épaisseur des lignes
plt.rcParams['legend.numpoints'] = 1
# un seul point dans la légende du graphique
plt.rc('axes', prop_cycle=(
    cycler('color', mglearn.plot_helpers.cm_cycle.colors) +
    cycler('linestyle', ['-', '-', "--", (0, (3, 3)), (0, (1.5, 1.5))])))
#défini le cycle de propriété des couleurs

np.set_printoptions(precision=3, suppress=True)
# pour l'affichage des talbeau numpy on garde juste 3 chiffres après la virgule

pd.set_option("display.max_columns", 8)
# le nombre max de colonnes
pd.set_option('display.precision', 2)
# défini les options d'affichage pour les dataframes pandas

### on importe les données du dataset ###
from sklearn.datasets import fetch_openml
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
# Extract features (X) and labels (y)
X, y = mnist.data, mnist.target


# We will use a subset of the data to speed up the computation
# on pourra l'enlever pour des meilleurs résultats après
np.random.seed(0)
# nombre pseudo aléatoire pour que l'expérience soit reproductible ! 
num_samples = 20000
indices = np.random.choice(len(X), num_samples)
X_subset = X.iloc[indices]
y_subset = y.iloc[indices]


### Split the data into training, validation, and test set ###
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_subset, y_subset, test_size=0.2, random_state=0)
# random_state = 0 pour garantir la reproductibilité des résultats 
X_train_mnist, X_val_mnist, y_train_mnist, y_val_mnist = train_test_split(X_train_mnist, y_train_mnist, test_size=0.2, random_state=0)
#cette fonction permet de couper sans mettre de préférence sur l'ordre ou quoi que ce soit
# Print the shapes


### On fait un algo en tri ###
tree = DecisionTreeClassifier(random_state=0)  # random state is used for tie-breaking
tree.fit(X_train_mnist, y_train_mnist)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train_mnist, y_train_mnist)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test_mnist, y_test_mnist)))


def afficher_image_index(index):
    # Afficher l'image choisie aléatoirement
    plt.imshow(X.loc[index].values.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title("Label: {}".format(y.loc[index]))
    plt.show()


def test_valeur_aleatoire():
    random_image_index = random.choice(X.index)
    afficher_image_index(random_image_index)
    new_observation = X.loc[random_image_index]
    prediction = tree.predict([new_observation])
    print(prediction)

# Choisir une image aléatoire de l'ensemble X

test_valeur_aleatoire()