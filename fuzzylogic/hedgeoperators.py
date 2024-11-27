import numpy as np

def concentration(mu, p):
    '''
    used to sharpen a fuzzy set, p > 1
    '''

    return np.power(mu, p)

def dilation(mu, q):
    '''
    used to blur a fuzzy set, q < 1
    '''

    return np.power(mu, q)


mu = np.array([0.2, 0.5, 0.8])

# Concentration with p > 1
concentrated_mu = concentration(mu, 2)
print("Concentrated:", concentrated_mu)

# Dilation with q < 1
dilated_mu = dilation(mu, 0.5)
print("Dilated:", dilated_mu)