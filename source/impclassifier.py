from decomposition import apply_imprecise, OVO, OVA, ECOC_dense, ECOC_sparse
from correct2 import unconditional_discounting
from recomposition import recompose
from collections.abc import Callable
import numpy as np

class DecompRecompImpreciseCART:
    def __init__(self, K: int, decompostion_scheme: str, confidence: float = 0.05) -> None:
        """
        Constructor of the classifier
        :param K: Number of classes
        :param decompostion_scheme: Decomposition scheme to use, either OVO, OVA, ECOC dense or ECOC sparse
        :param confidence: Confidence level for the imprecise classifier
        """
        self.K = K
        self.decomposition_scheme = OVO if decompostion_scheme == "OVO" else OVA if decompostion_scheme == "OVA" else ECOC_dense if decompostion_scheme == "ECOC_dense" else ECOC_sparse if decompostion_scheme == "ECOC_sparse" else None
        self.confidence = confidence
        self.classifiers = None
        self.decomps = None
        self.epsilons = None
        self.highest_density_intervals = None
    
    def fit(self, X_train, y_train) -> None:
        """
        Fit the classifier
        :param X_train: Training data
        :param y_train: Training labels
        """
        self.classifiers, self.decomps = self.decomposition_scheme(X_train, y_train)

    def predict(self, X_test) -> np.ndarray:
        """
        Predict the labels of the test data
        :param X_test: Test data
        :return: Predicted sets of classes
        """
        self.highest_density_intervals = apply_imprecise(X_test, self.classifiers, self.confidence)
        
        Y_set_pred = [set(range(self.K))]*len(X_test)
        for i in range(len(X_test)):
            alphas, betas = self.highest_density_intervals[i,::,0], self.highest_density_intervals[i,::,1]
            epsilons = unconditional_discounting(self.decomps, self.K, alphas, betas)
            for class_1 in range(self.K - 1):
                for class_2 in range(class_1, self.K):
                    rec = recompose(self.decomps, self.K, alphas, betas, epsilons, class_1, class_2)
                    if rec > 0:
                        Y_set_pred[i] = Y_set_pred[i] - {class_2}
        return Y_set_pred
    
