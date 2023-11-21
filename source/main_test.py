import matplotlib.pyplot as plt
from utils import generate_multiclass_data
import decomp
from sklearn.model_selection import train_test_split
from sklearn import tree
from scipy import stats
import numpy as np

np.random.seed(42)

# Parameters for each class (mean, std)
class_parameters = [
    ([1, 2], [1, 0.5]),
    ([4, 5], [1.25, 2]),
    ([0.25, 0.25], [0.5, 0.5]),
    ([-2, 1], [0.5, 1])
]

# Génération de données synthétiques multiclasse
num_samples_per_class = 100
X, y = generate_multiclass_data(num_samples_per_class, class_parameters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# Affichage graphique des données multiclasse
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Multiclass Synthetic Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
# Exemple de décomposition
classifiers, decomps = decomp.OVO(X_train,y_train)

C = decomp.apply_imprecise(X_test, classifiers, conf=0.05)

# Intervalles du classifieur 0
print(C[0])