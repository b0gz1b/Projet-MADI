import matplotlib.pyplot as plt
import numpy as np
from utils import generate_multiclass_data
import decomposition
from sklearn.model_selection import train_test_split
from sklearn import tree
from scipy import stats
import numpy as np
from correct2 import unconditional_discounting

np.random.seed(42)

# Parameters for each class (mean, std)
class_parameters = [
    ([1, 6], [1, 2.5]),
    ([4, -5], [1.25, 2]),
    ([-3.25, 0.25], [0.4, 0.8]),
    ([-6, 1], [0.5, 2])
]
K = 4
# Génération de données synthétiques multiclasse
num_samples_per_class = 100
X, y = generate_multiclass_data(num_samples_per_class, class_parameters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
x0 = 20
# Affichage graphique des données multiclasse avec des points pour les données d'entraînement et des croix pour les données de test, de la même couleur ! pour chaque classe, légendé
plt.figure()
for i in range(len(class_parameters)):
    plt.scatter(X_train[y_train==i,0], X_train[y_train==i,1], label="Classe {}".format(i))
    plt.scatter(X_test[y_test==i,0], X_test[y_test==i,1], marker="x", color="C{}".format(i))
# on ajoute juste un label sur le point x0 sans rien ajouter d'autre
plt.scatter(X_test[x0,0], X_test[x0,1], marker="x", color="black", label="Point x0")
plt.legend()
plt.show()
# Exemple de décomposition
classifiers, decomps = decomposition.OVO(X_train,y_train)

C, ks = decomposition.apply_imprecise(X_test, classifiers, conf=0.05)
print("Point {} : {} classe = {}".format(x0, X_test[x0], y_test[x0]))
for i in range(len(C)):
    # alpha in first column, beta in second column
    epsilons = unconditional_discounting(decomps, K, Alpha=C[i,::,0], Beta=C[i,::,1])
    print("Point {}, Classifier {} vs {} : [{};{}] eps = {} y hat = {} dens = {}".format(x0, decomps[i][0], decomps[i][1], C[i,x0,0], C[i,x0,1], epsilons[i], classifiers[i].predict(X_test[x0][np.newaxis]), ks[i,x0]))



