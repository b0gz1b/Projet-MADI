import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
from scipy.stats import norm

from impclassifier import DecompRecompImpreciseCART
from utils import experiment_results, discounted_accuracy, synthetic_datasets_generation

SEED = 0

np.random.seed(SEED)

# fetch dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
# data (as pandas dataframes) 

number_of_labels = len(np.unique(y_train))

# p = 15
# number_of_labels = 4
# shape_of_population = "ellipse"
# population_sizes = [50] * number_of_labels
# variability = "he"
# q = 0.75
# X, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)

ref65 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1.65)
ref80 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=2.2)
ref0 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1)
print(f"Reference u65 : {ref65[0]:.2f}")
print(f"Reference u80 : {ref80[0]:.2f}")
print(f"Reference weak : {ref0[0]:.2f}")

results = {'OVA' : {'time': 0, 'stats': None}, 'OVO' : {'time': 0, 'stats': None}, 'ECOC_dense' : {'time': 0, 'stats': None}, 'ECOC_sparse' : {'time': 0, 'stats': None}}

for decomp_strat in ["OVA", "OVO", "ECOC_dense", "ECOC_sparse"]:
    print(decomp_strat)
    cl=DecompRecompImpreciseCART(K=number_of_labels, decompostion_scheme=decomp_strat, confidence=0.05)
    cl.fit(X_train, y_train)

    start = time()
    Y_set_pred = cl.predict(X_test)
    end = time()
    print(f"Time : {end-start:.3f}s")
    results[decomp_strat]['time'] = end-start
    results[decomp_strat]['results'] = experiment_results(Y_set_pred, y_test)

    # noise = np.random.normal(0, 0.5, X_test.shape)
    # X_test_noisy = X_test + noise
    # Y_set_pred = cl.predict(X_test_noisy)
    # experiment_results(Y_set_pred, y_test)

def aux(std):
    confidence_level = 0.95
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    confidence_interval = z_score * (std / np.sqrt(len(y_test)))
    return confidence_interval

# Plot results

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(results.keys(), [results[decomp_strat]['time'] for decomp_strat in results.keys()])
plt.title('Time to predict (ROHD)')
plt.xlabel('Decomposition strategy')
plt.ylabel('Time (s)')
plt.savefig('time2.png')

#size
fig = plt.figure()
ax = fig.add_subplot(111)
#mean
ax.scatter(results.keys(), [results[decomp_strat]['results']['size'][0] for decomp_strat in results.keys()], marker='+')
#95% confidence interval
ax.errorbar(results.keys(), [results[decomp_strat]['results']['size'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['size'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5)
plt.title('Predicted set size (ROHD)')
plt.xlabel('Decomposition strategy')
plt.ylabel('Predicted set size')
plt.savefig('size2.png')

# Plot all accuracies on the same plot in the form of a point plot
fig = plt.figure()
ax = fig.add_subplot(111)
#mean
ax.scatter(results.keys(), [results[decomp_strat]['results']['u65'][0] for decomp_strat in results.keys()], marker='+', color='tab:blue')
ax.scatter(results.keys(), [results[decomp_strat]['results']['u80'][0] for decomp_strat in results.keys()], marker='+', color='tab:orange')
ax.scatter(results.keys(), [results[decomp_strat]['results']['u0'][0] for decomp_strat in results.keys()], marker='+', color='tab:green')
ax.scatter(results.keys(), [results[decomp_strat]['results']['sa'][0] for decomp_strat in results.keys()], marker='+', color='tab:red')
#95% confidence interval with the same color as the point
ax.errorbar(results.keys(), [results[decomp_strat]['results']['u65'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u65'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:blue')
ax.errorbar(results.keys(), [results[decomp_strat]['results']['u80'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u80'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:orange')
ax.errorbar(results.keys(), [results[decomp_strat]['results']['u0'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u0'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:green')
ax.errorbar(results.keys(), [results[decomp_strat]['results']['sa'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['sa'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:red')
#reference lines 
ax.axhline(y=ref65[0], color='tab:blue', linestyle='dotted',  label='u65 ref', alpha=0.5)
ax.axhline(y=ref80[0], color='tab:orange', linestyle='dotted',  label='u80 ref', alpha=0.5)
ax.axhline(y=ref0[0], color='tab:green', linestyle='dotted',  label='weak ref', alpha=0.5)

plt.title('Accuracy (ROHD)')
plt.xlabel('Decomposition strategy')
plt.ylabel('Accuracy')
plt.legend(['u65', 'u80', 'weak acc','set acc'])
plt.savefig('acc2.png')

# save results
import pickle
with open('results2.pkl', 'wb') as f:
    pickle.dump(results, f)