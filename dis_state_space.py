import numpy as np
from scipy.linalg import expm, inv

def dis_state_space(m, k, c, P, h, u0, udot0):
    n = len(P)
    
    #Metrix A, B, C, D
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])
    
    B = np.array([[0.0],
                  [1.0 / m]])
    
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    

    Ad = expm(A * h)
    
    I = np.eye(2)
    
    Bd = inv(A) @ (Ad - I) @ B
    
    x = np.array([[u0], 
                  [udot0]])
    
    u = np.zeros(n)
    u[0] = u0
    
    # Loop (Cd = C, Dd = D)
    for i in range(n - 1):
        u_input = np.array([[P[i]]])

        # Solve
        x_next = Ad @ x + Bd @ u_input
        y = C @ x_next + D @ u_input
        
        # Store the output
        u[i+1] = y[0, 0]
        
        # Another step
        x = x_next
        
    return u