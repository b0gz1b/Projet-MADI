from scipy.optimize import linprog
import numpy as np
from correct2 import unconditional_discounting

def recompose(decomp, K, Alpha, Beta, epsilons, class1, class2):

    J = len(decomp)

    c = np.zeros(K) # coef objectifs
    c[class1] = 1
    c[class2] = -1
    A_ub = np.zeros((2*J+2*K, K))
    b_ub = np.zeros(2*J+2*K)
    A_eq = np.ones(K)[np.newaxis]
    b_eq = np.ones(1)
    for i in range(2*J+2*K):
        if i < J: # Contraintes 18
            j = i
            b_ub[i] = Alpha[j] * epsilons[j]
            for k in range(K):
                if k in decomp[j][0]:
                    A_ub[i,k] = Alpha[j] - 1
                elif k in decomp[j][1]:
                    A_ub[i,k] = Alpha[j]
        elif i < 2*J: # Contraintes 19
            j = i - J
            b_ub[i] =  (1 - Beta[j]) * epsilons[j]
            for k in range(K):
                if k in decomp[j][0]:
                    A_ub[i,k] = 1 - Beta[j]
                elif k in decomp[j][1]:
                    A_ub[i,k] = -1 * Beta[j]
        elif i < 2*J+K: # Contraintes 21 (sur les p)
            k = i - 2*J
            A_ub[i,k] = -1
        elif i < 2*J+2*K: # Contraintes 21 (sur les p)
            k = i - 2*J - K
            A_ub[i,k] = 1
            b_ub[i] = 1
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    return res.fun


if __name__ == "__main__":
    decomp = [({0}, {1,2}), ({1}, {0,2}), ({2}, {0,1})]
    alpha = [0.4, 0.5, 0]
    beta =  [0.9, 0.7, 0.3]
    epsilons = unconditional_discounting(decomp, 3, alpha, beta)

    for i in range(len(decomp)):
        print("Classifer {} vs {} : [{};{}] eps = {}".format(decomp[i][0], decomp[i][1], alpha[i], beta[i], epsilons[i]))

    all_classes = set(range(3))
    for class1 in range(3):
        for class2 in range(3):
            if class1 != class2:
                rec = recompose(decomp, 3, alpha, beta, epsilons, class1, class2)
                print("Recomposition {} vs {} : {}".format(class1, class2, rec))
                if rec > 0:
                    print("Class {} is discarded".format(class2))
                    all_classes = all_classes - {class2}
    print("Remaining classes : {}".format(all_classes))