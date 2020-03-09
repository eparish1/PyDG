import numpy as np
import sys
import os

### script to make POD basis vectors

## determine number of blocks
n_blocks = 0
check = 0
while (check == 0):
  grid_str = 'DGgrid_block' + str(n_blocks) + '.npz'
  if (os.path.isfile(grid_str)):
    n_blocks += 1
  else:
     if (n_blocks == 0):
       print('Error, did not find ../DGgrid_block_ files')
       sys.exit()
     check = 1

ndata = 0
for j in range(0,n_blocks):
  sol_str = 'Solution_0/npsol_block' + str(j) + '_'  + str(0) + '.npz'
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
  end = 20000

#get strides
try:
  skip = int(sys.argv[3] )
except:
  skip = 1


try:
  sol_start = int(sys.argv[4] )
except:
  sol_start = 0

try:
  sol_end = int(sys.argv[5] )
except:
  sol_end = 20

try:
  sol_skip = int(sys.argv[6] )
except:
  sol_skip = 1

sys.stdout.write('Constructing POD basis functions' + '\n')
sys.stdout.write('Starting iteration = ' + str(start) + '\n')
sys.stdout.write('End iteration = ' + str(end) + '\n')
sys.stdout.write('Stride = ' + str(skip) + '\n')

for sol_folder in range(0,sol_end,sol_skip):
  nt = 0
  for i in range(0,end,skip):
    sys.stdout.write('On sol folder ' + str(sol_folder) + ' and sol number ' + str(i).zfill(5) + '\r')
    sys.stdout.flush()
    loc_array = np.zeros(0)
    for j in range(0,n_blocks):
      check = 0
      sol_str = 'Solution_' + str(sol_folder) + '/npsol_block' + str(j) + '_'  + str(i) + '.npz'
      if (os.path.isfile(sol_str)):
        check = 1
        #print('found ' + sol_str)
        data = np.load(sol_str)['a']
        loc_array = np.append(loc_array,data.flatten() )
        if (nt == 0):
          nparams += 1
        nt += 1
    if (check == 1):
      S = np.append(S,loc_array[:,None],axis=1)

## S  = N x Nt * Nsamp
S2 = np.reshape(S,(ndata,nt,nparams) )

S_spatial = np.reshape(S2,(ndata,nt*nparams) )
S_temporal = np.rollaxis(S2,1 )
### Start by making spatial basis
U,Lam,Z = np.linalg.svd(S_spatial,full_matrices=False) # Ns x Ks
### Now make temporal basis
Ut,Lam,Z = np.linalg.svd(S_temporal,full_matrices=False) # Nt x Kt

#project spatial:
test = np.einsum('ij,ijk->jk',U,S2)

#spacetime is in Ns x Nt
## Now make temporal basis via ST-HOSVD
# start by transforming snapshots
#VS = np.dot(U.transpose(),S)



np.savez('pod_basis_full',V=U,Lam=Lam)

## create truncated basis corresponding to 99%, 99.9%, and 99.99%

energy = np.sum(Lam)
nvec = np.size(Lam)
indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx] )
  if (rel_energy/energy >= 0.99 ):
    check = 1
    np.savez('pod_basis_99',V=U[:,0:indx])
    sys.stdout.write('99% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1


indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx] )
  if (rel_energy/energy >= 0.999 ):
    check = 1
    np.savez('pod_basis_999',V=U[:,0:indx])
    sys.stdout.write('99.9% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')

  else:
    indx += 1

indx = 1
check = 0
while (check == 0):
  rel_energy = np.sum(Lam[0:indx] )
  if (rel_energy/energy >= 0.9999 ):
    check = 1
    np.savez('pod_basis_9999',V=U[:,0:indx])
    sys.stdout.write('99.99% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
  else:
    indx += 1

