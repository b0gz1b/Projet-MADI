import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
from scipy.stats import norm

from impclassifier import DecompRecompImpreciseCART
from utils import experiment_results, discounted_accuracy, synthetic_datasets_generation, plot_2d_synthetic, plot_3d_synthetic

SEED = 0 # 0 pour le premier jeu de données, 1 pour le deuxième et 5 pour le troisième

np.random.seed(SEED)


if __name__ == "__main__":
    is_synthetic = False
    # data (as pandas dataframes) 
    if not is_synthetic:
        # fetch dataset 
        optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
        X = optical_recognition_of_handwritten_digits.data.features.to_numpy()
        y = optical_recognition_of_handwritten_digits.data.targets.to_numpy().flatten()
        number_of_labels = len(np.unique(y))
    else:
        p = 2
        number_of_labels = 4
        shape_of_population = "ellipse"
        population_sizes = [60] * number_of_labels
        variability = "he"
        q = 0.6
        X, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)
        if p==2:
            plot_2d_synthetic(X, y, distributions, title=f"{shape_of_population.capitalize()} {'heteroscedastic' if variability == 'he' else 'homoscedastic'}, q={q}", file_path=f"out/{shape_of_population}_{variability}_q{q}_{p}d.png")
        elif p==3:
            plot_3d_synthetic(X, y, distributions, title=f"{shape_of_population.capitalize()} {'heteroscedastic' if variability == 'he' else 'homoscedastic'}, q={q}", file_path=f"out/{shape_of_population}_{variability}_q{q}_{p}d.png")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=SEED)

    ref65 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1.65)
    ref80 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=2.2)
    ref0 = discounted_accuracy([set(range(number_of_labels))], [0], alpha_discount=1)
    print(f"Reference u65 : {ref65[0]:.2f}")
    print(f"Reference u80 : {ref80[0]:.2f}")
    print(f"Reference weak : {ref0[0]:.2f}")

    results = {'OVA' : {'time': 0, 'stats': None}, 
            'OVO' : {'time': 0, 'stats': None},
            'ECOC_dense' : {'time': 0, 'stats': None}, 
            'ECOC_sparse' : {'time': 0, 'stats': None}}
    names  = ['OVA', 'OVO', 'dense', 'sparse']

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
    plt.title('Time to predict')
    plt.xlabel('Decomposition strategy')
    plt.ylabel('Time (s)')
    if is_synthetic:
        plt.savefig(f"out/time_{shape_of_population}_{variability}_q{q}_{p}d.png")
    else:
        plt.savefig(f"out/time_digits.png")

    #size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #mean
    ax.scatter(results.keys(), [results[decomp_strat]['results']['size'][0] for decomp_strat in results.keys()], marker='+')
    #95% confidence interval
    ax.errorbar(results.keys(), [results[decomp_strat]['results']['size'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['size'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5)
    plt.title('Predicted set size')
    plt.xlabel('Decomposition strategy')
    plt.ylabel('Predicted set size')
    if is_synthetic:
        plt.savefig(f"out/size_{shape_of_population}_{variability}_q{q}_{p}d.png")
    else:
        plt.savefig(f"out/size_digits.png")

    # Plot all accuracies on the different plot in the same figure in 2x2 grid
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.set_title('u65')
    ax2.set_title('u80')
    ax3.set_title('weak')
    ax4.set_title('set')
    # set golbal title
    fig.suptitle('Accuracy')
    #mean
    ax1.scatter(names, [results[decomp_strat]['results']['u65'][0] for decomp_strat in results.keys()], marker='+', color='tab:blue')
    ax2.scatter(names, [results[decomp_strat]['results']['u80'][0] for decomp_strat in results.keys()], marker='+', color='tab:orange')
    ax3.scatter(names, [results[decomp_strat]['results']['u0'][0] for decomp_strat in results.keys()], marker='+', color='tab:green')
    ax4.scatter(names, [results[decomp_strat]['results']['sa'][0] for decomp_strat in results.keys()], marker='+', color='tab:red')
    #95% confidence interval with the same color as the point
    ax1.errorbar(names, [results[decomp_strat]['results']['u65'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u65'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:blue')
    ax2.errorbar(names, [results[decomp_strat]['results']['u80'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u80'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:orange')
    ax3.errorbar(names, [results[decomp_strat]['results']['u0'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['u0'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:green')
    ax4.errorbar(names, [results[decomp_strat]['results']['sa'][0] for decomp_strat in results.keys()], yerr=[aux(results[decomp_strat]['results']['sa'][1]) for decomp_strat in results.keys()], fmt='none', capsize=5, color='tab:red')
    #reference lines 
    ax1.axhline(y=ref65[0], color='tab:blue', linestyle='dotted',  label='u65 ref', alpha=0.5)
    ax2.axhline(y=ref80[0], color='tab:orange', linestyle='dotted',  label='u80 ref', alpha=0.5)
    ax3.axhline(y=ref0[0], color='tab:green', linestyle='dotted',  label='weak ref', alpha=0.5)
    plt.tight_layout()
    if is_synthetic:
        plt.savefig(f"out/accuracy_{shape_of_population}_{variability}_q{q}_{p}d.png")
    else:
        plt.savefig(f"out/accuracy_digits.png")

    # save results
    import pickle
    if is_synthetic:
        with open(f"out/results_{shape_of_population}_{variability}_q{q}_{p}d.pkl", 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(f"out/results_digits.pkl", 'wb') as f:
            pickle.dump(results, f)