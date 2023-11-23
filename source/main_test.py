import numpy as np
from utils import synthetic_datasets_generation, plot_2d_synthetic, plot_3d_synthetic, discounted_accuracy, set_accuracy
from sklearn.model_selection import train_test_split
from impclassifier import DecompRecompImpreciseCART
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo 
import itertools

np.random.seed(0)

# Parameters
p = 3
number_of_labels = 4
shape_of_population = "ellipse"
population_sizes = [100] * number_of_labels
variability = "he"
q = 0.80
X, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# Affichage graphique des données multiclasse avec des points pour les données d'entraînement et des croix pour les données de test, de la même couleur ! pour chaque classe, légendé
if p==2:
    plot_2d_synthetic(X, y, distributions)
elif p==3:
    plot_3d_synthetic(X, y, distributions)

# cl_ref = DecisionTreeClassifier()
# cl_ref.fit(X_train, y_train)

cl = DecompRecompImpreciseCART(K=p, decompostion_scheme="OVO", confidence=0.05)

cl.fit(X_train, y_train)

ref65 = discounted_accuracy([set(range(p))], [0], alpha_discount=1.65)
ref80 = discounted_accuracy([set(range(p))], [0], alpha_discount=2.2)
print(f"Reference u65 : {ref65[0]:.2f}±{ref65[1]:.2f}")
print(f"Reference u80 : {ref80[0]:.2f}±{ref80[1]:.2f}")

# Train

print("Accuracy on the training set")

Y_set_pred = cl.predict(X_train)

mean_pred_size = np.mean([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
std_pred_size = np.std([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
print(f"Mean predicted set size : {mean_pred_size:.2f}±{std_pred_size:.2f}")
u65, s65 = discounted_accuracy(Y_set_pred, y_train, alpha_discount=1.65)
u80, s80 = discounted_accuracy(Y_set_pred, y_train, alpha_discount=2.2)
mean_sa, std_sa = set_accuracy(Y_set_pred, y_train)
print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")
print(f"Set accuracy : {mean_sa:.2f}±{std_sa:.2f}")

# Test
print("Accuracy on the test set")

Y_set_pred = cl.predict(X_test)

mean_pred_size = np.mean([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
std_pred_size = np.std([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
print(f"Mean predicted set size : {mean_pred_size:.2f}±{std_pred_size:.2f}")
u65, s65 = discounted_accuracy(Y_set_pred, y_test, alpha_discount=1.65)
u80, s80 = discounted_accuracy(Y_set_pred, y_test, alpha_discount=2.2)
mean_sa, std_sa = set_accuracy(Y_set_pred, y_test)
print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")
print(f"Set accuracy : {mean_sa:.2f}±{std_sa:.2f}")

# Noise test
print("Accuracy on the test set with noise")

noise = np.random.normal(0, 0.5, X_test.shape)

X_test_noisy = X_test + noise
Y_set_pred = cl.predict(X_test_noisy)

if p==2:
    plot_2d_synthetic(X_test_noisy, y_test, distributions)
elif p==3:
    plot_3d_synthetic(X_test_noisy, y_test, distributions)

mean_pred_size = np.mean([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
std_pred_size = np.std([len(Y_set_pred[i]) for i in range(len(Y_set_pred))])
print(f"Mean predicted set size : {mean_pred_size:.2f}±{std_pred_size:.2f}")
u65, s65 = discounted_accuracy(Y_set_pred, y_test, alpha_discount=1.65)
u80, s80 = discounted_accuracy(Y_set_pred, y_test, alpha_discount=2.2)
mean_sa, std_sa = set_accuracy(Y_set_pred, y_test)
print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")
print(f"Set accuracy : {mean_sa:.2f}±{std_sa:.2f}")






# Handwritten Digits
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 
  
# metadata 
metadata = optical_recognition_of_handwritten_digits.metadata
  
# variable information 
print(optical_recognition_of_handwritten_digits.variables) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
X_train, X_test, y_train, y_test = X_train.values.tolist(), X_test.values.tolist(), list(itertools.chain.from_iterable(y_train.values.tolist())), list(itertools.chain.from_iterable(y_test.values.tolist()))
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

for decomp_strat in ["OVA", "OVO", "ECOC_dense", "ECOC_sparse"]:

    print(decomp_strat+"\n")
    cl=DecompRecompImpreciseCART(K=10, decompostion_scheme=decomp_strat, confidence=0.05)
    cl.fit(X_train, y_train)

    Y_set_pred = cl.predict(X_test)
    u65, s65 = cl.discounted_accuracy(Y_set_pred, y_test, alpha_discount=1.65)
    u80, s80 = cl.discounted_accuracy(Y_set_pred, y_test, alpha_discount=2.2)
    sa, err_sa = cl.set_accuracy(Y_set_pred, y_test)
    print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
    print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")
    print(f"Set accuracy u80 : {sa:.2f}±{err_sa:.2f}")