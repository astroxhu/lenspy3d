import numpy as np
import matplotlib.pyplot as plt

def sellmeier(lamb_nm, A1, A2, A3, B1, B2, B3):
    lamb = lamb_nm/1e3
    λ2 = lamb**2
    n2 = 1 + (A1 * λ2 / (λ2 - B1**2)) + (A2 * λ2 / (λ2 - B2**2)) + (A3 * λ2 / (λ2 - B3**2))
    return np.sqrt(n2)


