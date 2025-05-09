import numpy as np
import matplotlib.pyplot as plt

FRAUNHOFER_WAVELENGTHS = {
    'h': 404.7,
    'g': 435.8,
    "F'": 434.1,
    'F': 486.1,
    'e': 546.1,
    'd': 587.6,
    'D': 589.3,
    'C': 656.3,
    'r': 768.2,
}

def sellmeier(lamb_nm, A1, A2, A3, B1, B2, B3):
    lamb = lamb_nm/1e3
    λ2 = lamb**2
    n2 = 1 + (A1 * λ2 / (λ2 - B1**1)) + (A2 * λ2 / (λ2 - B2**1)) + (A3 * λ2 / (λ2 - B3**1))
    return np.sqrt(n2)

def find_index_in_ni(fraunhofer_key):
    prefixed = 'n' + fraunhofer_key
    try:
        return ni.index(prefixed)
    except ValueError:
        return -1  # or None if preferred

S_FPL53 = [9.83532327E-01,	6.95688140E-02,	1.11409238E+00,	4.92234955E-03,	1.93581091E-02,	2.64275294E+02]

ni = ["n2325",	"n1970",	"n1530",	"n1129",	"nt",	"ns",	"nA'",	"nr",	"nC",	"nC'",	"nHe-Ne",	"nD",	"nd",	"ne",	"nF",	"nF'",	"nHe-Cd",	"ng",	"nh",	"ni"]

n_S_FPL53 = [1.425123,	1.427618,	1.430316,	1.432688,	1.433463,	1.4348,	1.435702,	1.436522,	1.437333,	1.43756,	1.437771,	1.438709,	1.43875,	1.439854,	1.441955,	1.442215,	1.444098,	1.444423,	1.446451,	1.44986]


A1, A2, A3, B1, B2, B3 = [Ai for Ai in S_FPL53]

lamb_nm = np.linspace(400,770,371)

n_fit = sellmeier(lamb_nm, A1, A2, A3, B1, B2, B3)

Nlines=len(FRAUNHOFER_WAVELENGTHS)
print(Nlines)
lamblist = list(FRAUNHOFER_WAVELENGTHS.values())
nlist = [n_S_FPL53[find_index_in_ni(key)] for key in FRAUNHOFER_WAVELENGTHS]
fig=plt.figure()
ax=plt.subplot(111)
ax.plot(lamb_nm,n_fit)
ax.scatter(lamblist,nlist)
plt.show()
