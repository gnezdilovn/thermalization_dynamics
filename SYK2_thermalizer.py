import numpy as np
import functools as ft
import more_itertools
from more_itertools import distinct_permutations as idp
from scipy.linalg import expm

# down -- ground state
# up -- excited state
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

# psi(k, n) returns a list of states with k excitations among n qubits
# 0 symbolically represents the ground state
# 1 represents the excited state
# the ground state is given by psi(0, n)
def psi(k, n):
# listvec is a list of 0 and 1 
# where the position in the list corresponds to the qubit that is either in 0 or 1 state  
    listvec = [0] * (n - k) + [1] * k
    f = []
# the loop in idp(listvec) iterates all distinct permutations in listvec
# v is the list of 0 and 1 corresponding to a given permutation
    for i in idp(listvec):
        v = list(i)
        for j in range(len(v)):
            if v[j] == 0:
                v[j] = down # '0' in v is changed to 'down' vector         
            else:
                v[j] = up # '1' in v is changed to 'up' vector
# ft.reduce(np.kron, v) computes the Kronecker product between the elements of v     
        vec = ft.reduce(np.kron, v)
        f.extend([vec])
    return f

# perm_diag(n) returns a list of lists of \sigma^y_i, where i = 1, 2 ,..., n
def perm_diag(n):
# 'Y' represents \sigma^y operator 
    listvec = ['Y']  + ['I'] * (n - 1) 
    L1 = []
    for i in idp(listvec):
        v = list(i)   
        L1.extend([v])
    return L1  

# H_0(n, omega_min, omega_max) returns the frequencies of the qubits and 
# the initial Hamiltonian of the system  \sum_i 0.5 * \omega_i  sigma^y_i, 
# omega_min -- minimum frequency of the qubits
# omega_max -- maximum frequency of the qubits
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

# perm_SYK2(n) returns a list of lists of c_i^\dag c_j for i > j, where i = 1, 2 ,..., n
# where the fermionic operators c_i^\dag, c_i are mapped onto qubits using Jordan-Wigner transform
def perm_SYK2(n):
# listvec is a list of the operators 'P',  'M', and 'I' 
# where the position in the list corresponds to the qubit on which the operator acts on
# 'P' represents \sigma^+ operator acting onto the jth qubit, where j is the position of 'P' in the listvec
# 'M' represents \sigma^- operator acting onto the jth qubit, where j is the position of 'M' in the listvec
# 'I' represents the identity operator acting onto the jth qubit, where j is the position of 'I' in the listvec
    listvec = ['P', 'M']  + ['I'] * (n - 2)  
    L1 = [] # -- list of lists of terms that enter the Hamilonian
    L2 = [] # -- Hermitian conjugate of L1
    k = 0
# the loop in idp(listvec) iterates all distinct permutations in listvec
# v is the list of 'P', 'M', and 'I' corresponding to a given permutation
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
# we remove the elements of L1 that correspond to the Hermitian conjugate to avoid double counting          
        if k > 1:
            for m in range(2, k+1):
                if v == L2[m-2]:
                    L1.remove(v)
# the operator 'M' in each internal list of L1 positioned closest to 1 becomes the 'ZM' operator  
# the same holds for the 'P' and 'PZ' operators
# 'ZM' represents sigma^z \sigma^- operator
# 'PZ' represents sigma^+ \sigma^z operator
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
# we exchange the identity operators acting on qubits between the positions of the 'P' and 'M' operators 
# for the 'Z' operator.   
# 'Z' represents \sigma^z operator
    for k in range(len(L1)):
        for m in range(l[k]+1,len(L1[k])):
            if L1[k][m] == 'I':
                L1[k][m] = 'Z'
            elif L1[k][m] == 'P':
                break
            elif L1[k][m] == 'M':
                break
    return L1

# J_SYK2(n, J) returns a list of complex random numbers with the length of perm_SYK2(n) + perm_diag(n)  
def J_SYK2(n, J):
    varJ = J ** 2 / n 
    Js = [] 
    L1 = perm_SYK2(n)
    for i in range(len(L1)):    
        v = np.random.normal(loc = 0.0, scale = np.sqrt(0.5 * varJ), size = (1, 2)).view(np.complex)[0][0]
        Js.extend([v])
    return Js  

# H_SYK2(n, couplings, mu) returns 2d array for the Hamiltonian 
# H = \sum_{i>j}^n J_{ij} c_i^\dag c_j + h.c 
# for a given realization of couplings
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
# we exchange the string variables 'ZM', 'PZ', 'P', 'M', 'Z', and 'I' in L1[i] with the 2 x 2 matrices         
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
# we generate the Hamiltonian                
        H =  Js[k-1] * ft.reduce(np.kron, v) + np.conjugate(Js[k-1]) * ft.reduce(np.kron, vconj)
        H_sum = H_sum + H
        #for i in range(0,dim[0]):
        #    for j in range(0,dim[1]):
        #        if H_sum[i][j] - np.conjugate(H_sum[j][i]) != 0.:
        #            print('Hermicity check fails!')     
    return H_sum

# evolved_state(n, t, H) returns the ground state evolved with the Hamiltonian H
def evolved_state(n, t, H):
    gs = psi(0, n)[0]
    U = expm(- 1j * t * H)
    f = np.dot(U, gs)
    return f

# p(n, t, states, H) returns the occupation probability 
# p_k(\tau) = \sum_i |< k i | \psi(\tau) >|^2, 
# where | k i > is a state with occupation k and i sums over degenerate states
def p(n, t, states, H):
    pop_sum = 0.
    fs = states
    es = evolved_state(n, t, H)
    for i in range(len(fs)):
        overlap = np.dot(np.conjugate(fs[i]), es)
        pop = np.abs(overlap) ** 2
        pop_sum = pop_sum + pop
    return pop_sum

# we generate the data for a total spin of the system S, variance of S, and occupation probabilities 

# the parameters: 
# num -- number of qubits
# Jc -- square root of the variance of random couplings J_{ij}
# nr -- number of realizations
# omega_min -- minimum frequency of the qubit
# omega_max -- maximum frequency of the qubit
# t_min -- mininum time
# t_max -- maximum time
# nt -- number of time points

def SYK2(num, Jc, nr, omega_min, omega_max, t_min, t_max, nt):

    time = np.linspace(t_min, t_max, nt) 
    omegas = H_0(num, omega_min, omega_max)[0]
    H_qubits = H_0(num, omega_min, omega_max)[1]

    Lk = [ [0] * nr for t in range(len(time))] 
    Lk2 = [ [0] * nr for t in range(len(time))] 
    Lp = [ [0] * nr for t in range(len(time))] 
    S = [ [0] * nr for t in range(len(time))] 
    S2 = [ [0] * nr for t in range(len(time))] 
    SS2 = [ [0] * nr for t in range(len(time))]

    for j in range(nr):
    # printing the ongoing realization    
        print(j)
        Js = J_SYK2(num, Jc)
        H2 = H_SYK2(num, Js) + H_qubits
        for t in range(len(time)):
            ks = []
            ps = []
            for k in range(num+1):
                fs = psi(k, num)
                ks.append(k)
                ps.append(p(num, time[t], fs, H2))    
            Lk[t][j] = ks
            Lk2[t][j] = [q ** 2 for i, q in enumerate(ks)]
            Lp[t][j] = ps
            S[t][j] = - num / 2 + np.array(Lk[t][j]) @ np.array(Lp[t][j]) 
            S2[t][j] = np.array(Lk2[t][j]) @ np.array(Lp[t][j]) 
            SS2[t][j] =  S2[t][j] - ( S[t][j] + num / 2  ) ** 2

    # averaging over realizations: 
    S0 = np.array([- num / 2 for i in range(len(time))])  
    pav = np.sum(np.array(Lp), 1) / nr
    Sav = np.sum(np.array(S), 1)  / nr
    S2av = np.sum(np.array(S2), 1)  / nr
    VarS = S2av  - (Sav - S0) ** 2

    # saving the data:
    np.save('data/time.npy', time, allow_pickle = True)
    np.save('data/omegas_N={}_omega_min={}_omega_max={}.npy.npy'.format(num, omega_min, omega_max), omegas, allow_pickle = True)
    np.save('data/p_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), Lp, allow_pickle = True)
    np.save('data/pav_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), pav, allow_pickle = True)
    np.save('data/S_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), S, allow_pickle = True)
    np.save('data/SS2_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), SS2, allow_pickle = True)
    np.save('data/Sav_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), Sav, allow_pickle = True)
    np.save('data/S2av_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), S2av, allow_pickle = True)
    np.save('data/VarS_N={}_nr={}_omega_min={}_omega_max={}.npy'.format(num, nr, omega_min, omega_max), VarS, allow_pickle = True)
    # the data is saved into 'data/...'