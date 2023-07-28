import numpy as np
import functools as ft
import more_itertools
from more_itertools import distinct_permutations as idp
from scipy.linalg import expm
import random

# down -- ground state of a single qubit
# up -- excited state of a single qubit

down = np.array([1j, 1]) / np.sqrt(2)
up = np.array([-1j, 1]) / np.sqrt(2)

# 2 x 2 matrices acting in a single qubit space:
one = np.array([[1, 0], [0, 1]])
sy = np.array([[0, - 1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]]) 
sy = np.array([[0, - 1j], [1j, 0]])
plus = np.array([[0, 1], [0, 0]])
minus = np.array([[0, 0], [1, 0]])
mplus = plus @ sz
mminus = sz @ minus

def gs(n): 
    v = [0] * n
    for j in range(len(v)):
        if v[j] == 0:
            v[j] = down        
        else:
            v[j] = up     
    f = ft.reduce(np.kron, v)
    return f

def psi(n): 
    f = []
    for k in range(n+1):
        listvec = [0] * (n - k) + [1] * k
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

def energ_sp(n, omega_min, omega_max):
    f = np.zeros(n)
    for j in range(n):
        #v = (j + 1) ** 2 * omega
        f[j] = random.uniform(omega_min, omega_max)#v
    return f  

def perm_diag(n): 
    listvec = ['Y']  + ['I'] * (n - 1) 
    L1 = []
    for i in idp(listvec):
        v = list(i)   
        L1.extend([v])
    return L1  

def H_0(n, omegas):
    dim = (2 ** n, 2 ** n)
    f_sum = np.zeros(dim)
    L1 = perm_diag(n)
    for i in range(len(L1)):
        v = L1[i]
        for j in range(len(v)):
            if v[j] == 'Y':
                v[j] = sy  
            else:
                v[j] = one        
        f = 0.5 * omegas[i] * ft.reduce(np.kron, v) 
        f_sum = f_sum + f
    return f_sum

def energ_mb(n, omegas):
    fs = psi(n)
    ls = []
    es = []
    H = H_0(n, omegas)
    for l in range(len(fs)):
        ls.append(l)
        energ = np.conjugate(fs[l]) @ H @ fs[l]
        if np.abs(np.imag(energ)) > 1e-15:
            print('error: complex energies')
        es.append(np.real(energ))
    return ls, es
        
def perm_SYK2(n):
    listvec = ['P', 'M']  + ['I'] * (n - 2)  
    L1 = [] 
    L2 = []
    k = 0
    for i in idp(listvec):
        k += 1
        v = list(i)
        vconj = [0] * len(listvec)        
        for j in range(len(listvec)):
            if v[j] == 'P':
                vconj[j] = 'M'        
            elif v[j] == 'M':
                vconj[j] = 'P'   
            else:
                vconj[j] = 'I'    
        L1.extend([v])
        L2.extend([vconj])         
        if k > 1:
            for m in range(2, k+1):
                if v == L2[m-2]:
                    L1.remove(v)
    l = []    
    for k in range(len(L1)):
        for m in range(len(L1[k])):
            if L1[k][m] == 'P':
                L1[k][m] = 'PZ'
                l.extend([m])
                break
            elif L1[k][m] == 'M':
                L1[k][m] = 'ZM'
                l.extend([m])
                break  
    for k in range(len(L1)):
        for m in range(l[k]+1,len(L1[k])):
            if L1[k][m] == 'I':
                L1[k][m] = 'Z'
            elif L1[k][m] == 'P':
                break
            elif L1[k][m] == 'M':
                break
    return L1
 
def J_SYK2(n, J):
    varJ = J ** 2 / n 
    Js = [] 
    L1 = perm_SYK2(n)
    for i in range(len(L1)):    
        v = np.random.normal(loc = 0.0, scale = np.sqrt(0.5 * varJ), size = (1, 2)).view(np.complex)[0][0]
        Js.extend([v])
    return Js  

def H_SYK2(n, couplings):
    dim = (2 ** n, 2 ** n)
    H_sum = np.zeros(dim)
    L1 = perm_SYK2(n)
    Js = couplings
    k = 0
    for i in range(len(L1)):
        k += 1
        v = L1[i]        
        vconj = [0] * len(v)        
        for j in range(len(v)):
            if v[j] == 'ZM':
                v[j] = mminus
                vconj[j] = mplus
            elif v[j] == 'P':
                v[j] = plus
                vconj[j] = minus
            elif v[j] == 'PZ':
                v[j] = mplus 
                vconj[j] = mminus
            elif v[j] == 'M':
                v[j] = minus
                vconj[j] = plus
            elif v[j] == 'Z':
                v[j] = sz
                vconj[j] = sz
            else:
                v[j] = one
                vconj[j] = one                 
        H =  Js[k-1] * ft.reduce(np.kron, v) + np.conjugate(Js[k-1]) * ft.reduce(np.kron, vconj)
        H_sum = H_sum + H
        #for i in range(0,dim[0]):
        #    for j in range(0,dim[1]):
        #        if H_sum[i][j] - np.conjugate(H_sum[j][i]) != 0.:
        #            print('Hermicity check fails!')     
    return H_sum

def ev_state(n, t, H):
    U = expm(- 1j * t * H)
    f = np.dot(U, gs(n))
    return f

def p(n, t, H_0, V):
    fs = psi(n)
    es = []
    ps = []
    H_tot = H_0 + V
    psi_t = ev_state(n, t, H_tot)
    for l in range(len(fs)):
        energ = np.conjugate(fs[l]) @ H_0 @ fs[l]
        if np.abs(np.imag(energ)) > 1e-15:
            print('error: complex energies')
        es.append(np.real(energ))
        pop = np.abs(np.dot(np.conjugate(fs[l]), psi_t)) ** 2
        ps.append(pop)
    es_order = []
    ps_order = []    
    for l in range(len(fs)):    
        es_order.append(sorted(zip(es, ps))[l][0])
        ps_order.append(sorted(zip(es, ps))[l][1])
    return es_order, ps_order

def p_th(n, b, omegas):
    es = energ_mb(n, omegas)[1]
    ps = []
    for l in range(len(es)):
        pop = np.exp(- b * es[l])
        ps.append(pop)
    Z = np.sum(ps)
    for l in range(len(es)):
        ps[l] = ps[l] / Z 
    es_order = []
    ps_order = []
    for l in range(len(es)):
        es_order.append(sorted(zip(es, ps))[l][0])
        ps_order.append(sorted(zip(es, ps))[l][1])
    return es_order, ps_order

# the parameters: 
# num -- number of qubits
# Jc -- square root of the variance of random couplings J_{ij}
# nr -- number of realizations
# [omega_min, omega_max] -- defines the interval for the single-particle energies
# do -- levels' spacing
# t_min -- mininum time
# t_max -- maximum time
# nt -- number of time points

def SYK2(num, Jc, nr, omega_min, omega_max, t_min, t_max, nt):

    time = np.linspace(t_min, t_max, nt) 
    sp_levels = energ_sp(num, omega_min, omega_max)
    H_qubits = H_0(num, sp_levels)
    energ = sorted(energ_mb(num, sp_levels)[1])


    Le = [[0] * nr for t in range(len(time))]
    Le2 = [[0] * nr for t in range(len(time))]
    Lp = [[0] * nr for t in range(len(time))] 
    E = [[0] * nr for t in range(len(time))] 
    E2 = [[0] * nr for t in range(len(time))] 
    EE2 = [[0] * nr for t in range(len(time))]

    for j in range(nr):
        print(j)
        Js = J_SYK2(num, Jc)
        V = H_SYK2(num, Js)
        for t in range(len(time)):
            Le[t][j] = p(num, time[t], H_qubits, V)[0]
            delta = np.array(Le[t][j]) - np.array(energ)
            if delta.any() != 0.:
                print('error: the energies do not coincide')
            Le2[t][j] = [q ** 2 for i, q in enumerate(Le[t][j])]    
            Lp[t][j] = p(num, time[t], H_qubits, V)[1]
            E[t][j] = np.array(Le[t][j]) @ np.array(Lp[t][j])
            E2[t][j] = np.array(Le2[t][j]) @ np.array(Lp[t][j]) 
            EE2[t][j] =  E2[t][j] - E[t][j] ** 2
 
    pav = np.sum(np.array(Lp), 1) / nr
    Eav = np.sum(np.array(E), 1)  / nr
    E2av = np.sum(np.array(E2), 1)  / nr
    VarE = E2av  - Eav ** 2

    np.save('data/time.npy', time, allow_pickle = True)
    np.save('data/sp_levels_N={}_omega_min={}_omega_max={}.npy'.format(num, omega_min, omega_max), sp_levels, allow_pickle = True)
    np.save('data/energ_N={}_omega_min={}_omega_max={}.npy'.format(num, omega_min, omega_max), energ, allow_pickle = True)
    np.save('data/es_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), Le, allow_pickle = True)
    np.save('data/p_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), Lp, allow_pickle = True)
    np.save('data/pav_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), pav, allow_pickle = True)
    np.save('data/E_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), E, allow_pickle = True)
    np.save('data/Eav_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), Eav, allow_pickle = True)
    np.save('data/EE2_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), EE2, allow_pickle = True)
    np.save('data/E2av_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), E2av, allow_pickle = True)
    np.save('data/VarE_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), VarE, allow_pickle = True)
    # the data is saved into 'data/...'