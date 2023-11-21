from scipy.optimize import minimize
import numpy as np

def unconditional_discounting(Alpha, Beta, K):

    def f_obj(e):
        return e.sum()
    
    def contraintes(e, p, alpha, beta):

        contraintes_e = [1 - e_ij for e_ij in e]
        contraintes_p = [1 - p_i for p_i in p]
        contraintes_alpha = [((1-alpha[i][j])*p[i] - alpha[i][j]*p[j] - e[i][j]*(-alpha[i][j])) for i in range(K) for j in range(i+1,K)]
        contraintes_beta = [((beta[i][j]-1)*p[i] + beta[i][j]*p[j] - e[i][j]*(beta[i][j]-1)) for i in range(K) for j in range(i+1,K)]

        return contraintes_e + contraintes_p + contraintes_alpha + contraintes_beta
    
    alpha = Alpha
    beta = Beta
    p0 = [0.2] * K
    e0 = np.array([[0,0.1,0.1,0.1],[0,0,0.1,0.1],[0,0,0,0.1],[0]*4])

    c_eq = {'type': 'eq', 'fun': lambda p: sum(p) - 1}
    c_ineq = {'type': 'ineq', 'fun': contraintes, 'args': (p0, alpha, beta)}

    res = minimize(f_obj, e0, constraints=[c_eq, c_ineq])

    print("Valeur minimale trouvée :", res.fun)
    print("Valeur de la variable à minimiser (e) :", res.x)

    return res.fun, res.x

