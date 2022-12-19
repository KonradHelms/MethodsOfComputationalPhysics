import numpy as np
from numpy import linalg
from tqdm import tqdm

print("------- Parameters -------")
epsilon = 0.2
mu = 0.1
delta_x = 0.4
delta_t = 0.1
n_x = 130
n_t = 2000
L = n_x*delta_x
print(f"   * epsilon = {epsilon}")
print(f"   * mu = {mu}")
print(f"   * n_x = {n_x}")
print(f"   * n_t = {n_t}")
print(f"   * delta_x = {delta_x}")
print(f"   * L = {L}")
print(f"   * delta_t = {delta_t}")
print("--------------------------")

# a) 
# is a pen and paper exercise 

# b)
print("Part b):\n")
def kdev_u_2(epsilon:float,mu:float,delta_x:float,delta_t:float,u_1:np.array,exercise:str)->np.array:
    if exercise=="b":
        u_1[0] = 1 # boundary conditions given in the ex.
        u_1[len(u_1)-1] = 0 # boundary condtions given in the ex.
    if exercise=="c":
        u_1[0] = 0 # boundary conditions given in the ex.
        u_1[len(u_1)-1] = 0 # boundary condtions given in the ex.

    u_j_plus_1 = np.roll(u_1,-1)
    u_j_plus_1[-1] = u_1[len(u_1)-1]    # this is what I understand from the boundary conditions
    u_j_minus_1 = np.roll(u_1,1)
    u_j_minus_1[0] = u_1[0]             # -"-
    u_j_plus_2 = np.roll(u_j_plus_1,-1)
    u_j_plus_2[-1] = u_j_plus_1[len(u_j_plus_1)-1] # -"-
    u_j_minus_2 = np.roll(u_j_minus_1,1)
    u_j_minus_2[0] = u_j_minus_1[0]     # -"-     
    u_new = u_1 - epsilon/6 * delta_t/delta_x * np.multiply(u_j_plus_1 + u_1 + u_j_minus_1,u_j_plus_1 - u_j_minus_1) - mu/2 * delta_t/(delta_x**3) * (u_j_plus_2 + 2*u_j_minus_1 - 2*u_j_plus_1 - u_j_minus_2)
    return u_new


def kdev_n_plus_1(epsilon:float,mu:float,delta_x:float,delta_t:float,u_n_minus_1:np.array,u_n:np.array,exercise:str)->np.array:
    if exercise=="b":
        u_n[0] = 1 # boundary conditions given in the ex.
        u_n[len(u_n)-1] = 0 # boundary condtions given in the ex.
    if exercise=="c":
        u_n[0] = 0 # boundary conditions given in the ex.
        u_n[len(u_n)-1] = 0 # boundary condtions given in the ex.
    
    u_j_plus_1 = np.roll(u_n,-1)
    u_j_plus_1[-1] = u_n[len(u_n)-1] 
    u_j_minus_1 = np.roll(u_n,1)
    u_j_minus_1[0] = u_n[0]
    u_j_plus_2 = np.roll(u_j_plus_1,-1)
    u_j_plus_2[-1] = u_j_plus_1[len(u_j_plus_1)-1]
    u_j_minus_2 = np.roll(u_j_minus_1,1)
    u_j_minus_2[0] = u_j_minus_1[0]
    u_new = u_n_minus_1 - epsilon/3 * delta_t/delta_x * np.multiply(u_j_plus_1 + u_n + u_j_minus_1, u_j_plus_1 - u_j_minus_1) - mu * delta_t/(delta_x**3) * (u_j_plus_2 + 2*u_j_minus_1 - 2*u_j_plus_1 - u_j_minus_2)
    return u_new


def initial_condition(x:np.array)->np.array:
    return 0.5*(1-np.tanh((x-25)/5))

def stability_condition(delta_x:float,delta_t:float,epsilon:float,mu:float,u:np.array)->float:
    return delta_t/delta_x * (epsilon * np.linalg.norm(u) + 4*mu/(delta_x**2))

def run_simulation(epsilon:float,mu:float,n:int,x:np.array,delta_x:float,delta_t:float,condition:str)->np.array:
    print("Running simulation ...")
    if condition=="b":
        init_condition = initial_condition
    if condition=="c":
        init_condition = colliding_solitons_initial_condition
    u = init_condition(x)
    us = [u]
    u = kdev_u_2(epsilon,mu,delta_x,delta_t,u,condition)
    us.append(u)
    for i in tqdm(range(n)):
        u = kdev_n_plus_1(epsilon,mu,delta_x,delta_t,us[i],u,condition)
        us.append(u)
    print("... done")
    return np.array(us)

def stability_analysis(delta_x:float,delta_t:float,epsilon:float,mu:float,us:np.array)->np.array:
    print("Runnig stability analysis ...")
    stabilities = []
    for u in tqdm(us):
        stab = stability_condition(delta_x,delta_t,epsilon,mu,u)
        stabilities.append(stab)
    print("... done")
    return np.array(stabilities)

x = np.arange(0,L,delta_x)
u = run_simulation(epsilon,mu,n_t,x,delta_x,delta_t,"b") 
np.save("sheet4-3-us.npy",u)
stabilities = stability_analysis(delta_x,delta_t,epsilon,mu,u)  # stabilities for some reason don't look correct, they are just above 1
print(f"Stability test sucess: {bool(set(stabilities[stabilities<=1]))}")
print("--------------------------")

# c)
print("Part c):\n")

def colliding_solitons_initial_condition(x:np.array)->np.array:
    return 0.8 * (1 - np.tanh(3*x/12 - 3)**2) + 0.3 * (1 - np.tanh(4.5*x/26 - 4.5)**2)
n_t = 10000
u_c = run_simulation(epsilon,mu,n_t,x,delta_x,delta_t,"c") 
np.save("sheet4-3-us-part-c.npy",u_c)
stabilities = stability_analysis(delta_x,delta_t,epsilon,mu,u_c)
print(f"Stability test sucess: {bool(set(stabilities[stabilities<=1]))}")