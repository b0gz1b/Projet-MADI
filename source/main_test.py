from time import time
import numpy as np
from utils import synthetic_datasets_generation, plot_2d_synthetic, plot_3d_synthetic, discounted_accuracy, experiment_results
from sklearn.model_selection import train_test_split
from impclassifier import DecompRecompImpreciseCART

np.random.seed(0)

# Parameters
p = 2
number_of_labels = 4
shape_of_population = "ellipse"
population_sizes = [50] * number_of_labels
variability = "he"
q = 0.75
X, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# Affichage graphique des données multiclasse avec des points pour les données d'entraînement et des croix pour les données de test, de la même couleur ! pour chaque classe, légendé
if p==2:
    plot_2d_synthetic(X, y, distributions)
elif p==3:
    plot_3d_synthetic(X, y, distributions)

# cl_ref = DecisionTreeClassifier()
# cl_ref.fit(X_train, y_train)

ref65 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1.65)
ref80 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=2.2)
ref0 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1)
print(f"Reference u65 : {ref65[0]:.2f}±{ref65[1]:.2f}")
print(f"Reference u80 : {ref80[0]:.2f}±{ref80[1]:.2f}")
print(f"Reference weak : {ref0[0]:.2f}±{ref0[1]:.2f}")

# Test
for decomp_strat in ["OVA", "OVO", "ECOC_dense", "ECOC_sparse"]:
    print(decomp_strat)
    cl=DecompRecompImpreciseCART(K=number_of_labels, decompostion_scheme=decomp_strat, confidence=0.05)
    cl.fit(X_train, y_train)

    start = time()
    Y_set_pred = cl.predict(X_test)
    end = time()
    print(f"Time : {end-start:.3f}s")
    experiment_results(Y_set_pred, y_test)