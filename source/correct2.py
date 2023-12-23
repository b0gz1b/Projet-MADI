from scipy.optimize import linprog
import numpy as np

def unconditional_discounting(decomp, K, Alpha, Beta):

    J = len(decomp)

    c = np.concatenate((np.ones(J), np.zeros(K))) # coef objectifs
    A_ub = np.zeros((4*J+2*K, J+K))
    b_ub = np.zeros(4*J+2*K)
    A_eq = np.concatenate((np.zeros(J), np.ones(K)))[np.newaxis]
    b_eq = np.ones(1)
    for i in range(4*J+2*K):
        if i < J: # Contraintes 18
            j = i
            A_ub[i,j] = -1 * Alpha[j]
            for k in range(K):
                if k in decomp[j][0]:
                    A_ub[i,J+k] = Alpha[j] - 1
                elif k in decomp[j][1]:
                    A_ub[i,J+k] = Alpha[j] 
        elif i < 2*J: # Contraintes 19
            j = i - J
            A_ub[i,j] = Beta[j] - 1
            for k in range(K):
                if k in decomp[j][0]:
                    A_ub[i,J+k] = 1 - Beta[j]
                elif k in decomp[j][1]:
                    A_ub[i,J+k] = -1 * Beta[j]
        elif i < 3*J: # Contraintes 21 (sur les epsilon)
            j = i - 2*J
            A_ub[i,j] = -1
        elif i < 4*J: # Contraintes 21 (sur les epsilon)
            j = i - 3*J
            A_ub[i,j] = 1
            b_ub[i] = 1
        elif i < 4*J+K: # Contraintes 21 (sur les p)
            k = i - 4*J
            A_ub[i,J+k] = -1
        elif i < 4*J+2*K: # Contraintes 21 (sur les p)
            k = i - 4*J - K
            A_ub[i,J+k] = 1
            b_ub[i] = 1
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    return res.x[:J]


if __name__ == "__main__":
    decomp = [({0}, {1,2}), ({1}, {0,2}), ({2}, {0,1})]
    alpha = [0.4, 0.7, 0.1]
    beta =  [0.9, 1, 0.3]
    epsilons = unconditional_discounting(decomp, 3, alpha, beta)

    for i in range(len(decomp)):
        print("Classifer {} vs {} : [{};{}] eps = {}".format(decomp[i][0], decomp[i][1], alpha[i], beta[i], epsilons[i]))