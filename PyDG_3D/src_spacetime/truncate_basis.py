import numpy as np

basis = np.load('POD_basis.npz')

lam = basis['Lam']

lam_total = np.sum(lam)

e_frac = 0.
i = 1
while (e_frac <= 0.99):
  e_frac = sum(lam[0:i])/lam_total
  i += 1

print('We require ' + str(i) + ' basis to capture 99%' )

np.savez('POD_basis_trunctate.npz',V=basis['V'][:,0:i])
