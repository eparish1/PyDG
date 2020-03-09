import numpy as np
import sys
import os

### script to make POD basis vectors

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
  sol_end = 1

try:
  sol_skip = int(sys.argv[6] )
except:
  sol_skip = 1

sys.stdout.write('Constructing POD basis functions' + '\n')
sys.stdout.write('Starting iteration = ' + str(start) + '\n')
sys.stdout.write('End iteration = ' + str(end) + '\n')
sys.stdout.write('Stride = ' + str(skip) + '\n')
nparams = 0
for sol_folder in range(0,sol_end,sol_skip):
  nt = 0
  for i in range(0,end,skip):
    sys.stdout.write('On sol folder ' + str(sol_folder) + ' and sol number ' + str(i).zfill(5) + '\r')
    sys.stdout.flush()
    loc_array = np.zeros(0)
    for j in range(0,n_blocks):
      check = 0
      #sol_str = 'Solution_' + str(sol_folder) + '/npsol_block' + str(j) + '_'  + str(i) + '.npz'
      sol_str = 'npsol_block' + str(j) + '_'  + str(i) + '.npz'

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
Us,Lams,Z = np.linalg.svd(S_spatial,full_matrices=False) # Ns x Ks
#Us = Us[:,0:10]
### Now make temporal basis
Ut,Lamt,Z = np.linalg.svd(S_temporal[:,:,0],full_matrices=False) # Nt x Kt
#Ut = Ut[:,0:20]

#UF = U[:,:,None,None]*Ut[None,None,:,:]
#
#
#

def construct_st_basis(Us,Ut,Lams,Lamt,tol_s,tol_t):
  energy_s  = np.sum(Lams)
  energy_t  = np.sum(Lamt)
  nvec_s = np.size(Lams)
  nvec_t = np.size(Lamt)

  indx_s = 1
  indx_t = 1
  check = 0
  ## Determine number of spatial basis
  while (check == 0):
    rel_energy_s = np.sum(Lams[0:indx_s] )
    if (rel_energy_s/energy_s >= tol_s):
      check = 1
    else:
      indx_s += 1
  ## Determine number of temporal basis
  check = 0
  while (check == 0):
    rel_energy_t = np.sum(Lamt[0:indx_t] )
    if (rel_energy_t/energy_t >= tol_t):
      check = 1
    else:
      indx_t += 1

  ## Create tensor product basis
  #UF = U[:,0:indx_s,None,None]*Ut[None,None,:,0:indx_t]
  sys.stdout.write(str(tol_s) + ' of spatial energy = ' + str(indx_s) + '/' + str(nvec_s) + ' spatial basis vectors \n')
  sys.stdout.write(str(tol_t) + ' of temporal energy = ' + str(indx_t) + '/' + str(nvec_t) + ' temporal basis vectors \n')
  np.savez('st_pod_basis_' + str(tol_s) + '_' + str(tol_t),Vs=Us[:,0:indx_s],Vt=Ut[:,0:indx_t])

  

construct_st_basis(Us,Ut,Lams,Lamt,0.97,0.97)
construct_st_basis(Us,Ut,Lams,Lamt,0.99,0.99)
construct_st_basis(Us,Ut,Lams,Lamt,0.999,0.999)











