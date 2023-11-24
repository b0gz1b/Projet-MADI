import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from time import time

from impclassifier import DecompRecompImpreciseCART
from utils import experiment_results, discounted_accuracy

SEED = 0

np.random.seed(SEED)

# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features.to_numpy()
y = optical_recognition_of_handwritten_digits.data.targets.to_numpy().flatten()
number_of_labels = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=SEED)

ref65 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1.65)
ref80 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=2.2)
print(f"Reference u65 : {ref65[0]:.2f}±{ref65[1]:.2f}")
print(f"Reference u80 : {ref80[0]:.2f}±{ref80[1]:.2f}")

for decomp_strat in ["OVA", "OVO", "ECOC_dense", "ECOC_sparse"]:
    print(decomp_strat)
    cl=DecompRecompImpreciseCART(K=number_of_labels, decompostion_scheme=decomp_strat, confidence=0.05)
    cl.fit(X_train, y_train)

    start = time()
    Y_set_pred = cl.predict(X_test)
    end = time()
    print(f"Time : {end-start:.3f}s")
    experiment_results(Y_set_pred, y_test)

    # noise = np.random.normal(0, 0.5, X_test.shape)
    # X_test_noisy = X_test + noise
    # Y_set_pred = cl.predict(X_test_noisy)
    # experiment_results(Y_set_pred, y_test)
    