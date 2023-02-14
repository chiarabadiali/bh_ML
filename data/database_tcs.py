import numpy as np
import pandas as pd
from itertools import product

# Module with Bethe Heitler cross-sections
import cross_section

#----------------------------------------------------------------
# Range of the physical parameter
#----------------------------------------------------------------

# Z atomic number
min_z, max_z, N_z = 1, 100, 1000
z_range = np.linspace(min_z, max_z, N_z)

# k photon energy
min_k, max_k, N_k = 2.1e0, 2.e4, 1000
k_range = np.logspace(np.log10(min_k), np.log10(max_k), N_k)

#----------------------------------------------------------------
# Data frame with the total cross-section
#----------------------------------------------------------------

# Range of variables
data = dict()
vars = list(product(z_range, k_range))

# Build the variables and the target
data['Z'] = np.array([elem[0] for elem in vars])
data['k'] = np.array([elem[1] for elem in vars])

# Initialise the array
N = int(N_z * N_k)
data['f'] = np.zeros([N])

# Fill the array
for i in range(N):
    data['f'][i] = cross_section.bh_cs(vars[i][0], vars[i][1])

df = pd.DataFrame(data)

# Store the result in a csv file
df.to_csv('data/BH_tcs.csv', index=False, header=False)

print('job done')