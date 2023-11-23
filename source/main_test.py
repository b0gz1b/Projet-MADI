import matplotlib.pyplot as plt
import numpy as np
from utils import generate_multiclass_data
from sklearn.model_selection import train_test_split
from impclassifier import DecompRecompImpreciseCART


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
num_samples_per_class = 50
X, y = generate_multiclass_data(num_samples_per_class, class_parameters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# Affichage graphique des données multiclasse avec des points pour les données d'entraînement et des croix pour les données de test, de la même couleur ! pour chaque classe, légendé
plt.figure()
for i in range(len(class_parameters)):
    plt.scatter(X_train[y_train==i,0], X_train[y_train==i,1], label="Classe {}".format(i))
    plt.scatter(X_test[y_test==i,0], X_test[y_test==i,1], marker="x", color="C{}".format(i))
plt.legend()
plt.show()

cl = DecompRecompImpreciseCART(K=K, decompostion_scheme="OVA", confidence=0.05)

cl.fit(X_train, y_train)

Y_set_pred = cl.predict(X_test)

u65, s65 = cl.discounted_accuracy(Y_set_pred, y_test, alpha_discount=1.65)
u80, s80 = cl.discounted_accuracy(Y_set_pred, y_test, alpha_discount=2.2)
print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")