import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

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

def generate_binary_vectors(p):
    # Generate all possible combinations of {-1, 1} for a given length p
    binary_vectors = np.array(list(np.ndindex((2,) * p)))

    # Convert 0 to -1 in the binary vectors
    binary_vectors[binary_vectors == 0] = -1

    return binary_vectors

def synthetic_datasets_generation(p: int, 
                                  number_of_labels: int, 
                                  shape_of_population: str,
                                  population_sizes: list[int],
                                  variability: str,
                                  q: int) -> np.ndarray:
    """
    Generate synthetic datasets
    :param number_of_features: Number of features
    :param number_of_labels: Number of labels
    :param shape_of_population: Shape of the population, either "sphere" or "ellipse"
    :param population_sizes: Sizes of the sub-populations
    :param variability: Variability of the population, either "ho" or "he", for homoscedastic or heteroscedastic respectively
    :param q: Quantile of the Chi squared distribution with p degrees of freedom
    :return: Synthetic dataset
    """
    # covariance matrix
    if shape_of_population == "sphere":
        if variability == "ho":
            cov = np.eye(p)
        elif variability == "he":
            cov = [(2*np.random.random()) * np.eye(p) for _ in range(number_of_labels)]
        else:
            raise ValueError("variability must be either 'ho' or 'he'")
    elif shape_of_population == "ellipse":
        if variability == "ho":
            cov = random_positive_semi_definite_matrix(p)
        elif variability == "he":
            cov = [random_positive_semi_definite_matrix(p) for _ in range(number_of_labels)]
        else:
            raise ValueError("variability must be either 'ho' or 'he'")
    else:
        raise ValueError("shape_of_population must be either 'sphere' or 'ellipse'")
    
    dataset = np.zeros((np.sum(population_sizes), p))
    y = np.concatenate([np.full((population_sizes[i], 1), i) for i in range(number_of_labels)])
    distributions = []
    # Root sub-population
    mu_mr = np.zeros(p)
    if variability == "ho":
        dataset[0:population_sizes[0]] = np.random.multivariate_normal(mu_mr, cov, size=population_sizes[0])
    elif variability == "he":
        dataset[0:population_sizes[0]] = np.random.multivariate_normal(mu_mr, cov[0], size=population_sizes[0])
    
    # generate the delta coefficients
    vectors = generate_binary_vectors(p)
    omegas = vectors[np.random.choice(len(vectors), size=number_of_labels, replace=False)]

    # Other sub-populations
    for i in range(1, number_of_labels):
        if variability == "ho":
            delta = 2 * np.sqrt(np.linalg.eig(cov)[0] * chi2.ppf(q, p))
            dist = omegas[i] * delta
            distributions.append((mu_mr + dist, cov))
            dataset[sum(population_sizes[:i]):sum(population_sizes[:i])+population_sizes[i]] = np.random.multivariate_normal(mu_mr + dist, cov, size=population_sizes[i])
        elif variability == "he":
            a1 = 0.8 * np.sqrt(np.linalg.eig(cov[0])[0] * chi2.ppf(q, p))
            a2 = 0.8 * np.sqrt(np.linalg.eig(cov[i])[0] * chi2.ppf(q, p))
            delta = a1 + a2
            dist = omegas[i] * delta
            distributions.append((mu_mr + dist, cov))
            dataset[sum(population_sizes[:i]):sum(population_sizes[:i])+population_sizes[i]] = np.random.multivariate_normal(mu_mr + dist, cov[i], size=population_sizes[i])

    return dataset, y.flatten(), distributions

if __name__ == "__main__":
    np.random.seed(0)
    # Parameters
    p = 2
    number_of_labels = 3
    shape_of_population = "sphere"
    population_sizes = [40, 40, 40]
    variability = "ho"
    q = 0.80

    # Generate synthetic dataset
    dataset, y, distributions = synthetic_datasets_generation(p, number_of_labels, shape_of_population, population_sizes, variability, q)
    # Plot the dataset
    plt.figure()
    for i in range(number_of_labels):
        plt.scatter(dataset[y==i,0], dataset[y==i,1], label="Classe {}".format(i))
    plt.legend()

    plt.show()