from scipy.optimize import minimize
import numpy as np

def unconditional_discounting(decomp, Alpha, Beta):

    J = len(decomp)

    def f_obj(e):
        return sum(e)
    
    def contraintes(e, p, alpha, beta):

        contraintes_e = [1 - e_j for e_j in e]
        contraintes_p = [1 - p_k for p_k in p]
        contraintes_alpha = [((1-alpha[j])*sum([p[i] for i in decomp[j][0]]) - alpha[j]*sum([p[i] for i in decomp[j][1]]) - e[j]*(-alpha[j])) for j in range(J)]
        contraintes_beta = [((beta[j]-1)*sum([p[i] for i in decomp[j][0]]) + beta[j]*sum([p[i] for i in decomp[j][1]]) - e[j]*(beta[j]-1)) for j in range(J)]

        return contraintes_e + contraintes_p + contraintes_alpha + contraintes_beta

    p0 = [1/J]*J
    e0 = [1/J]*J

    c_eq = {'type': 'eq', 'fun': lambda p: sum(p) - 1}
    c_ineq = {'type': 'ineq', 'fun': contraintes, 'args': (p0, Alpha, Beta)}

    res = minimize(f_obj, e0, constraints=[c_eq, c_ineq])
    print(res.message)
    print("Valeur minimale trouvée :", res.fun)
    print("Valeur de la variable à minimiser (e) :", res.x)

    return res.fun, res.x

