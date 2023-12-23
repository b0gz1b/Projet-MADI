import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Dict, List, Tuple

def generate_multiclass_data(num_samples, class_parameters):
    data = []
    labels = []
    
    for class_label, (mean, std) in enumerate(class_parameters):
        class_data = np.random.normal(loc=mean, scale=std, size=(num_samples, 2))
        class_labels = np.full((num_samples, 1), class_label)
        
        data.append(class_data)
        labels.append(class_labels)
    
    # Concatenate data and labels for each class
    data = np.vstack(data)
    labels = np.vstack(labels)
    
    return data, labels.flatten()

def random_positive_semi_definite_matrix(p: int) -> np.ndarray:
    """
    Generate a random positive semi-definite matrix using Cholesky decomposition
    :param p: Dimension of the matrix
    :return: Random positive semi-definite matrix
    """
    A = np.random.rand(p, p)
    B = np.dot(A, A.transpose())
    return B

def g_covariance(p, sphere=True, eigv_large=1.0):
    # Generate a random matrix and perform QR decomposition
    random_matrix = np.random.normal(size=(p, p))
    Q, _ = np.linalg.qr(random_matrix)

    # Generate eigenvalues based on the specified conditions
    if sphere:
        eigvalues = np.full(p, np.random.uniform(0, eigv_large))
    else:
        eigvalues = np.sort(np.random.uniform(0, eigv_large, p))[::-1]

    # Calculate the covariance matrix Sigma
    H = Q * np.sqrt(eigvalues)
    Sigma = np.dot(H, H.T)
    eigvectors = np.linalg.eig(Sigma)[1]
    return {'eigvals': eigvalues, 'eigvectors': eigvectors,'Sigma': Sigma}

def generate_unique_random_arrays(n, m):
    random_arrays = np.empty((n, m), dtype=int)

    for i in range(n):
        while True:
            # Generate a random array of size m with values -1 or +1
            random_array = np.random.choice([-1, 1], size=m)

            # Check if the array is unique
            if not np.any(np.all(random_array == random_arrays[:i], axis=1)):
                break  # Exit the loop if the array is unique

        random_arrays[i] = random_array

    return random_arrays

def synthetic_datasets_generation(p: int, 
                                  number_of_labels: int, 
                                  shape_of_population: str,
                                  population_sizes: List[int],
                                  variability: str,
                                  q: int,
                                  eigv_large: float=1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic datasets
    :param number_of_features: Number of features
    :param number_of_labels: Number of labels
    :param shape_of_population: Shape of the population, either "sphere" or "ellipse"
    :param population_sizes: Sizes of the sub-populations
    :param variability: Variability of the population, either "ho" or "he", for homoscedastic or heteroscedastic respectively
    :param q: Quantile of the Chi squared distribution with p degrees of freedom
    :param eigv_large: Largest eigenvalue of the covariance matrix
    
    :return: Synthetic dataset with the corresponding labels. Also returns the distributions used to generate the dataset
    """
    if number_of_labels > 2**p:
        raise ValueError("number_of_labels must be less than 2**p where p is the number of features")
    if q < 0 or q > 1:
        raise ValueError("q must be strictly between 0 and 1")

    if variability == "ho":
        cov = g_covariance(p, sphere=True if shape_of_population=="sphere" else False, eigv_large=eigv_large)
    elif variability == "he":
        cov = [g_covariance(p, sphere=True if shape_of_population=="sphere" else False, eigv_large=eigv_large) for _ in range(number_of_labels)]
    
    dataset = np.zeros((np.sum(population_sizes), p))
    y = np.concatenate([np.full((population_sizes[i], 1), i) for i in range(number_of_labels)])
    distributions = []
    # Root sub-population
    mu_mr = np.zeros(p)
    if variability == "ho":
        dataset[0:population_sizes[0]] = np.random.multivariate_normal(mu_mr, cov["Sigma"], size=population_sizes[0])
        distributions.append((mu_mr, cov))
    elif variability == "he":
        dataset[0:population_sizes[0]] = np.random.multivariate_normal(mu_mr, cov[0]["Sigma"], size=population_sizes[0])
        distributions.append((mu_mr, cov[0]))
    
    # generate the delta coefficients
    omegas = generate_unique_random_arrays(number_of_labels, p)

    # Other sub-populations
    for i in range(1, number_of_labels):
        if variability == "ho":
            delta = 2 * np.sqrt(cov["eigvals"] * chi2.ppf(q, p))
            dist = omegas[i] * delta
            distributions.append((mu_mr + dist, cov))
            dataset[sum(population_sizes[:i]):sum(population_sizes[:i])+population_sizes[i]] = np.random.multivariate_normal(mu_mr + dist, cov["Sigma"], size=population_sizes[i])
        elif variability == "he":
            a1 = 0.8 * np.sqrt(cov[0]["eigvals"] * chi2.ppf(q, p))
            a2 = 0.8 * np.sqrt(cov[1]["eigvals"] * chi2.ppf(q, p))
            delta = a1 + a2
            dist = omegas[i] * delta
            distributions.append((mu_mr + dist, cov[i]))
            dataset[sum(population_sizes[:i]):sum(population_sizes[:i])+population_sizes[i]] = np.random.multivariate_normal(mu_mr + dist, cov[i]["Sigma"], size=population_sizes[i])

    return dataset, y.flatten(), distributions

def plot_2d_synthetic(dataset, y, distributions, title=None, file_path=None):
    fig, ax = plt.subplots()
    for i in range(len(distributions)):
        ax.scatter(dataset[y==i,0], dataset[y==i,1], label="Class {}".format(i))
        v, vec = distributions[i][1]["eigvals"], distributions[i][1]["eigvectors"]
        angle = np.arctan2(vec[1, 0], vec[0, 0])
        std_devs = np.sqrt(v[0]), np.sqrt(v[1])
        c = patches.Ellipse((distributions[i][0][0], distributions[i][0][1]), 2 * std_devs[0], 2 * std_devs[1], angle=np.degrees(angle), fill=False, color="black", linestyle="dashed")
        ax.add_patch(c)
        c = patches.Ellipse((distributions[i][0][0], distributions[i][0][1]), 4 * std_devs[0], 4 * std_devs[1], angle=np.degrees(angle), fill=False, color="black", linestyle="dotted")
        ax.add_patch(c)
    ax.set_aspect('equal', 'box')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Synthetic dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()

def plot_3d_synthetic(dataset, y, distributions, title=None, file_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(distributions)):
        ax.scatter(dataset[y==i,0], dataset[y==i,1], dataset[y==i,2], label="Class {}".format(i))
    ax.set_aspect('equal', 'box')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Synthetic dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.legend()
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()

def discounted_accuracy(Y_set_pred, y_test, alpha_discount) -> float:
    """
    Compute the discounted accuracy of the classifier
    :param X_test: Test data
    :param y_test: Test labels
    :param alpha_discount: Discount factor
    :return: Discounted accuracy
    """
    discounted_accurracies = [(alpha_discount / len(Y_set_pred[i])) - ((alpha_discount - 1) / len(Y_set_pred[i])**2) if y_test[i] in Y_set_pred[i] else 0 for i in range(len(y_test))]
    return np.mean(discounted_accurracies), np.std(discounted_accurracies)

def set_accuracy(Y_set_pred, y_test) -> float:
    """
    Compute the set accuracy of the classifier
    :param X_test: Test data
    :param y_test: Test labels
    :return: Set accuracy
    """
    set_accurracies = [1 if y_test[i] in Y_set_pred[i] else 0 for i in range(len(y_test))]
    return np.mean(set_accurracies), np.std(set_accurracies)

def experiment_results(Y, y, verbose=True):
    u65, s65 = discounted_accuracy(Y, y, alpha_discount=1.65)
    u80, s80 = discounted_accuracy(Y, y, alpha_discount=2.2)
    u0, s0 = discounted_accuracy(Y, y, alpha_discount=1)
    sa, err_sa = set_accuracy(Y, y)
    mean_pred_size = np.mean([len(Y[i]) for i in range(len(Y))])
    std_pred_size = np.std([len(Y[i]) for i in range(len(Y))])
    if verbose:
        print(f"Predicted set size : {mean_pred_size:.2f}±{std_pred_size:.2f}")
        print(f"Discounted accuracy u65 : {u65:.2f}±{s65:.2f}")
        print(f"Discounted accuracy u80 : {u80:.2f}±{s80:.2f}")
        print(f"Weak accuracy: {u0:.2f}±{s0:.2f}")
        print(f"Set accuracy : {sa:.2f}±{err_sa:.2f}")
    return {'size': (mean_pred_size, std_pred_size), 'u65': (u65, s65), 'u80': (u80, s80), 'u0': (u0, s0), 'sa': (sa, err_sa)}

if __name__ == "__main__":
    np.random.seed(0)
    # Parameters
    p = 2
    number_of_labels = 4
    shape_of_population = "ellipse"
    population_sizes = [40] * number_of_labels
    variability = "he"
    q = 0.75

    # Generate synthetic dataset
    dataset, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)
    # Plot the dataset
    if p==2:
        plot_2d_synthetic(dataset, y, distributions)
    elif p==3:
        plot_3d_synthetic(dataset, y, distributions)