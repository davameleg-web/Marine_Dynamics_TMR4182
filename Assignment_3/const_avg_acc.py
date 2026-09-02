import numpy as np

def const_avg_acc(m, k, c, P, h, u0, udot0):
    n = len(P)
    u = np.zeros(n)
    udot = np.zeros(n)
    uddot = np.zeros(n)
    
    # Initial conditions
    u[0] = u0
    udot[0] = udot0
    
    # Initial acceleration from the equilibrium equation at time t = 0
    uddot[0] = (P[0] - c * udot0 - k * u0) / m
    
    # Effective stiffness (constant for linear SDOF)
    k_eff = (4.0 / (h**2)) * m + (2.0 / h) * c + k
    
    # Time loop
    for i in range(n - 1):
        # Assembling the right-hand side (RHS)
        rhs = (P[i+1] + 
               m*uddot[i] + 
               (4/h*m+c)*udot[i] +
               (4/(h**2)*m+2/h*c)*u[i])
        
        # 1. Calculation of the unknown displacement u_{i+1}
        u[i+1] = rhs / k_eff
        
        # 2. Calculation of the acceleration
        uddot[i+1] = (4.0 / (h**2)) * (u[i+1] - u[i] - udot[i] * h) - uddot[i]
        
        # 3. Calculation of the velocity
        udot[i+1] = udot[i] + 0.5 * (uddot[i] + uddot[i+1]) * h
        
    return u