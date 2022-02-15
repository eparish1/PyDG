import numpy as np
import sys
import os
### script to make POD basis vectors

tol = float(sys.argv[1])
## determine number of blocks
n_blocks = 0
check = 0
while (check == 0):
  grid_str = '../DGgrid_block' + str(n_blocks) + '.npz'
  if (os.path.isfile(grid_str)):
    n_blocks += 1
  else:
     if (n_blocks == 0):
       print('Error, did not find ../DGgrid_block_ files')
       sys.exit()
     check = 1

ndata = 0
for j in range(0,n_blocks):
  sol_str = 'npsol_block' + str(j) + '_'  + str(0) + '.npz'
  data = np.load(sol_str)['a']
  ndata += np.size(data)

## Create array for snapshots
S = np.zeros((ndata,0) )

# get starting iteration number
try:
  start = int( sys.argv[1] )
except:
  start = 0

# get final iteration number
try:
  end = int( sys.argv[2] )
except: 
  end = 200000

#get strides
try:
  skip = int(sys.argv[3] )
except:
  skip = 1

sys.stdout.write('Constructing POD basis functions')
sys.stdout.write('Starting iteration = ' + str(start) + '\n')
sys.stdout.write('End iteration = ' + str(end) + '\n')
sys.stdout.write('Stride = ' + str(skip) + '\n')


sys.stdout.write('Loading mass matrix'  + '\n')
Msqrt = []
Msqrtinv = []
M = []
for j in  range(0,n_blocks):
  grid = np.load('../DGgrid_block' + str(j) + '.npz')
  Msqrtloc = grid['Minv']
  shp = np.shape(Msqrtloc)
  ndofs = np.prod(np.shape(Msqrtloc)[0:4])
  Nel = np.shape(Msqrtloc)[-4::]
  Msqrtloc = np.reshape(Msqrtloc, (ndofs,ndofs,Nel[0],Nel[1],Nel[2],Nel[3]) )
  Msqrtloc = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis(Msqrtloc,2,0),3,1),4,2),5,3)
 
  Mloc = np.linalg.inv(Msqrtloc)
  Msqrtloc = np.linalg.cholesky(np.linalg.inv(Msqrtloc))

  #nsz = np.size(np.shape(Msqrtloc))
  #transposeCoords = np.array(range(0,nsz))
  #transposeCoords[0] = 1
  #transposeCoords[1] = 0
  #Msqrtloc = np.transpose(Msqrtloc,axes=transposeCoords)


  Msqrtlocinv = np.linalg.inv(Msqrtloc) 

  Mloc = np.rollaxis(np.rollaxis(Mloc,4,0),5,1)
  Msqrtloc = np.rollaxis(np.rollaxis(Msqrtloc,4,0),5,1)
  Msqrtlocinv = np.rollaxis(np.rollaxis(Msqrtlocinv,4,0),5,1)

  Mloc = np.reshape(Mloc,shp)
  Msqrtloc = np.reshape(Msqrtloc,shp)
  Msqrtlocinv = np.reshape(Msqrtlocinv,shp)

  M.append(Mloc)
  Msqrt.append(Msqrtloc)
  Msqrtinv.append(Msqrtlocinv)

for i in range(0,end,skip):
  print('On sol number ' + str(i) )
  loc_array = np.zeros(0)
  for j in range(0,n_blocks):
    check = 0
    sol_str = 'npsol_block' + str(j) + '_'  + str(i) + '.npz'
    if (os.path.isfile(sol_str)):
      check = 1
      #print('found ' + sol_str)
      data = np.load(sol_str)['a']
      data = np.sum(Msqrt[j][None]*data[:,None,None,None,None],axis=(5,6,7,8) )
      loc_array = np.append(loc_array,data.flatten() )
  if (check == 1):
    S = np.append(S,loc_array[:,None],axis=1)

S = S[:,:] #- base[:,None]
U,Lam,Z = np.linalg.svd(S,full_matrices=False)

### Compute t
N,K = np.shape(U)
for i in range(0,K):
  tmp = np.reshape(U[:,i],np.shape(data))
  tmp = np.sum(Msqrtinv[0][None]*tmp[:,None,None,None,None],axis=(5,6,7,8) )
  U[:,i] = tmp.flatten()


#np.savez('pod_basis_full',V=U,Lam=Lam)



## create truncated basis corresponding to 99%, 99.9%, and 99.99%
sol_str = 'npsol_block' + str(j) + '_'  + str(0) + '.npz'
totalEnergy = np.sum(Lam**2)
relativeEnergy = np.cumsum(Lam**2) / totalEnergy
K = np.size( relativeEnergy[relativeEnergy<tol] ) + 1
np.savez('pod_basis_' + str(tol) , V=U[:,0:K],relativeEnergy=relativeEnergy)

'''
nvec = np.size(Lam)
indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx]**2 )
  if (rel_energy/energy >= tolerance ):
    check = 1
    np.savez('pod_basis_95',V=U[:,0:indx])
    sys.stdout.write('95% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1



## create truncated basis corresponding to 99%, 99.9%, and 99.99%
sol_str = 'npsol_block' + str(j) + '_'  + str(0) + '.npz'
data = np.load(sol_str)
energy = np.sum(Lam**2)
nvec = np.size(Lam)
indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx]**2)
  if (rel_energy/energy >= 0.97 ):
    check = 1
    np.savez('pod_basis_97',V=U[:,0:indx])
    sys.stdout.write('97% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1



sol_str = 'npsol_block' + str(j) + '_'  + str(0) + '.npz'
data = np.load(sol_str)
energy = np.sum(Lam**2)
nvec = np.size(Lam)
indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx]**2 )
  if (rel_energy/energy >= 0.99 ):
    check = 1
    np.savez('pod_basis_99',V=U[:,0:indx])
    sys.stdout.write('99% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1


indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx]**2)
  if (rel_energy/energy >= 0.999 ):
    check = 1
    np.savez('pod_basis_999',V=U[:,0:indx])
    sys.stdout.write('99.9% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')

  else:
    indx += 1

indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx]**2)
  if (rel_energy/energy >= 0.9999 ):
    check = 1
    np.savez('pod_basis_9999',V=U[:,0:indx])
    sys.stdout.write('99.99% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1
'''
