import numpy as np
from numpy import linalg
from tqdm import tqdm

print("------- Parameters -------")
L = 1
K = 210
C = 900
rho = 2700
delta_x = 0.01
delta_t = 0.1
n_t = 10000
print(f"   * L = {L}")
print(f"   * K = {K}")
print(f"   * C = {C}")
print(f"   * rho = {rho}")
print(f"   * delta_x = {delta_x}")
print(f"   * delta_t = {delta_t}")
print(f"   * n_t = {n_t}")
print("--------------------------")


def setup_initial_conditions(L:float,x:np.array)->np.array:
    return np.array(np.sin(np.pi*x/L))


def ftcs_step(u_j:np.array,delta_t:float,delta_x:float,C:float,rho:float)->np.array:
    lambd = K/(C*rho)
    a = lambd*delta_t/(delta_x**2)
    u_n_j_plus_1 = np.roll(u_j,-1)
    u_n_j_plus_1[-1] = 0
    u_n_j_minus_1 = np.roll(u_j,1)
    u_n_j_minus_1[0] = 0
    return (1-2*a)*u_j+a*(u_n_j_minus_1+u_n_j_plus_1)


def crank_nicolson_step(u_j:np.array,delta_t:float,delta_x:float,C:float,rho:float)->np.array:
    lambd = K/(C*rho)
    a =lambd*delta_t/(2*delta_x**2)
    dim = u_j.shape[0]
    # create matrix A from the lecture
    A = np.identity(dim) + 2*a*np.identity(dim)
    A[0,0] = 1
    A[dim-1,dim-1] = 1
    A[np.eye(len(A),k=1,dtype="bool")] = -a
    A[np.eye(len(A),k=-1,dtype="bool")] = -a
    A[0,1] = 0
    A[dim-1,dim-2] = 0
    # create matrix B from the lecture 
    B = np.identity(dim) - 2*a*np.identity(dim)
    B[0,0] = 1
    B[dim-1,dim-1] = 1
    B[np.eye(len(B),k=1,dtype="bool")] = a
    B[np.eye(len(B),k=-1,dtype="bool")] = a
    B[0,1] = 0
    B[dim-1,dim-2] = 0
    A_inv = np.linalg.inv(A)
    return np.matmul(A_inv,B).dot(u_j)


def euler_backward_step(u_j:np.array,delta_t:float,delta_x:float,C:float,rho:float)->np.array:
    lambd = K/(C*rho)
    a = lambd*delta_t/(delta_x**2)
    dim = u_j.shape[0]
    A = np.identity(dim) + 2*a*np.identity(dim)
    A[0,0] = 1
    A[dim-1,dim-1] = 1
    A[np.eye(len(A),k=1,dtype="bool")] = -a
    A[np.eye(len(A),k=-1,dtype="bool")] = -a
    A[0,1] = 0
    A[dim-1,dim-2] = 0
    A_inv = np.linalg.inv(A)
    return A_inv.dot(u_j)


def dufort_frankel_step(u_n_minus_1_j:np.array,u_n_j:np.array,delta_t:float,delta_x:float,C:float,rho:float)->np.array:
    lambd = K/(C*rho)
    a = 2*lambd*delta_t/(delta_x**2)
    u_n_j_plus_1 = np.roll(u_n_j,-1)
    u_n_j_plus_1[-1] = 0
    u_n_j_minus_1 = np.roll(u_n_j,1)
    u_n_j_minus_1[0] = 0
    return (1-a)/(1+a) * u_n_minus_1_j + a/(1+a) * (u_n_j_plus_1 + u_n_j_minus_1) 


def analytical_solution(L:float,K:float,C:float,rho:float,t:float,x:np.array)->np.array:
    return np.sin(np.pi*x/L)*np.exp(-np.pi**2*K*t/(L**2*C*rho))


def error(L:float,delta_x:float,T:np.array,T_exact:np.array)->np.array:
    N_x = L/delta_x
    return 1/N_x * np.sum(np.abs(T-T_exact))


def run_simulation(L:float,delta_x:float,delta_t:float,n_t:int,C:float,rho:float,algorithm:str)->np.array:
    if algorithm == "ftcs":
        step = ftcs_step
    if algorithm == "cn":
        step = crank_nicolson_step
    if algorithm == "backward":
        step = euler_backward_step
    if algorithm == "dff":
        step = dufort_frankel_step
    T = setup_initial_conditions(L,x)
    Ts = [T]
    errs = [np.zeros_like(T)]
    if algorithm != "dff":
        for _ in tqdm(range(n_t)):
            T = step(T,delta_t,delta_x,C,rho)
            Ts.append(T)
    else:
        for i in tqdm(range(n_t)):
            if i<=1: # because we have to get u^n-1 and u^n to infer u^n+1, CN step was just arbitrary
                T = crank_nicolson_step(T,delta_t,delta_x,C,rho)
                Ts.append(T)
            else:
                T = dufort_frankel_step(Ts[i-1],T,delta_t,delta_x,C,rho)
                Ts.append(T)
    return np.array(Ts)


def run_error_analysis(delta_ts:list,file_trail:str,step:str):
    file_trail += "_"+step
    for delta_t in delta_ts:
        print(f"Calculating the error for delta_t = {delta_t} ...")
        t = 100
        n_t = int(t/delta_t)
        T = run_simulation(L,delta_x,delta_t,n_t,C,rho,step)
        t_idx = int(t/delta_t) # need this, as the array T contains the temp. distr. T at all times 
        T_exact = analytical_solution(L,K,C,rho,t,x)
        err = error(L,delta_x,T[t_idx],T_exact)
        print(f"the error is {err}")
        np.save(f"sheet4-ex2-{file_trail}-err-{delta_t}.npy",err)
        np.save(f"sheet4-ex2-{file_trail}-err-{delta_t}_T.npy",T[t_idx])
    np.save(f"T_exact_{file_trail}.npy",analytical_solution(L,K,C,rho,100,x))


# a)
x = np.arange(0,L,delta_x) # this will be used in several parts of this exercise

print("PART a):\n")
T = run_simulation(L,delta_x,delta_t,n_t,C,rho,"ftcs")
np.save("sheet4-ex2.npy",T)
print("--------------------------")

# b)
print("PART b):\n")
delta_ts = [0.001,0.01,0.1,0.3,0.5,0.6,0.65]
run_error_analysis(delta_ts,"part_b","ftcs")
print("--------------------------")

# c)
print("PART c):\n")
steps = ["cn","backward","dff"]
for step in steps:
    run_error_analysis(delta_ts,"part_c",step)
print("--------------------------")