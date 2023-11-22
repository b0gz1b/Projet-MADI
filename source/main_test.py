import matplotlib.pyplot as plt
import numpy as np
from utils import generate_multiclass_data
import decomp
from correct import unconditional_discounting

# # Parameters for each class (mean, std)
# class_parameters = [
#     ([1, 2], [1, 0.5]),
#     ([4, 5], [1.25, 2]),
#     ([0.25, 0.25], [0.5, 0.5]),
#     ([-2, 1], [0.5, 1])
# ]

# # Génération de données synthétiques multiclasse
# num_samples_per_class = 100
# X, y = generate_multiclass_data(num_samples_per_class, class_parameters)

# # Affichage graphique des données multiclasse
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
# plt.title('Multiclass Synthetic Data Distribution')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

# # Exemple de décomposition
# classifiers, decomps = decomp.ECOC_dense(X,y)
# print(decomps)
# print(classifiers)

decomp = [({0}, {1}), ({0}, {2}), ({0}, {3}), ({1}, {2}), ({1}, {3}), ({2}, {3})]
alpha = [0.6,0.6,0,0.6,0.6,0.6]
beta = [1,1,0.4,1,1,1]
unconditional_discounting(decomp, alpha, beta)