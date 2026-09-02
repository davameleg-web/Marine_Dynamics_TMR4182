import numpy as np

def runge_kutta4(m, k, c, P, h, u0, udot0):
    n = len(P)
    u = np.zeros(n)
    udot = np.zeros(n)
    
    u[0] = u0
    udot[0] = udot0
    
    # Define the system of first-order ODEs
    def dYdt(Y, P_val):
        y1, y2 = Y[0], Y[1]
        dy1 = y2
        dy2 = (P_val - c * y2 - k * y1) / m
        return np.array([dy1, dy2])
    
    for i in range(n - 1):
        Y_i = np.array([u[i], udot[i]])
        
        P_i = P[i]
        P_next = P[i+1]
        P_mid = 0.5 * (P_i + P_next) # this is approximation; the assignment does not allow passing an equation to the function.
        
        # Koefficients
        K1 = dYdt(Y_i, P_i)
        K2 = dYdt(Y_i + 0.5 * h * K1, P_mid)
        K3 = dYdt(Y_i + 0.5 * h * K2, P_mid)
        K4 = dYdt(Y_i + h * K3, P_next)
        
        Y_next = Y_i + (h / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)
        
        u[i+1] = Y_next[0]
        udot[i+1] = Y_next[1]
        
    return u