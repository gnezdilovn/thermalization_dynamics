import numpy as np
import functools as ft
import more_itertools
from more_itertools import distinct_permutations as idp
from scipy.linalg import expm
from scipy.special import binom

down = np.array([1j, 1]) / np.sqrt(2)
up = np.array([-1j, 1]) / np.sqrt(2)
one = np.array([[1, 0], [0, 1]])
sy = np.array([[0, - 1j], [1j, 0]])

# occupation probabilities at infinite temperature
def p_inf(k, n):
    f = 1. / 2. ** n * binom(n, k)
    return f

# occupation probabilities at finite temperature for degenerate qubits
def p_th_deg(k, b, n, omega):
    f = np.exp(- b * omega * k) / (1. + np.exp(- omega * b)) ** n * binom(n, k)
    return f

def psi(k, n):
    listvec = [0] * (n - k) + [1] * k
    f = []
    for i in idp(listvec):
        v = list(i)
        for j in range(len(v)):
            if v[j] == 0:
                v[j] = down         
            else:
                v[j] = up     
        vec = ft.reduce(np.kron, v)
        f.extend([vec])
    return f

def perm_diag(n):
    listvec = ['Y']  + ['I'] * (n - 1) 
    L1 = []
    for i in idp(listvec):
        v = list(i)   
        L1.extend([v])
    return L1  

def H_0(n, omega_min, omega_max):
    dim = (2 ** n, 2 ** n)
    f_sum = np.zeros(dim)
    L1 = perm_diag(n)
    omegas = np.linspace(omega_max, omega_min, n)
    for i in range(len(L1)):
        v = L1[i]
        for j in range(len(v)):
            if v[j] == 'Y':
                v[j] = sy  
            else:
                v[j] = one        
        f = 0.5 * omegas[i] * ft.reduce(np.kron, v) 
        f_sum = f_sum + f
    return omegas, f_sum

# occupation probabilities at finite temperature
def p_th(k, n, b, omegas):
    pop_sum = 0.
    ebh = expm(- b * H_0(n, min(omegas), max(omegas))[1]) 
    Z = np.trace(ebh)
    rho_th = ebh / Z
    fs = psi(k, n)
    for i in range(len(fs)):
        pop = np.conjugate(fs[i]) @ rho_th @ fs[i]
        pop_sum = pop_sum + pop
    return pop_sum