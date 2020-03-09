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
     check = 1

ndata = 0
for j in range(0,n_blocks):
  sol_str = 'Solution_0/npsol_block' + str(j) + '_'  + str(0) + '.npz'
  data = np.load(sol_str)['a']
  ndata += np.size(data[0])
nvars = np.shape(data)[0]

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


sys.stdout.write('Loading mass matrix'  + '\n')
Msqrt = []
Msqrtinv = []
M = []
for j in  range(0,n_blocks):
  grid = np.load('DGgrid_block' + str(j) + '.npz')
  Msqrtloc = grid['Minv']
  shp = np.shape(Msqrtloc)
  ndofs = np.prod(np.shape(Msqrtloc)[0:4])
  Nel = np.shape(Msqrtloc)[-4::]
  Msqrtloc = np.reshape(Msqrtloc, (ndofs,ndofs,Nel[0],Nel[1],Nel[2],Nel[3]) )
  Msqrtloc = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis(Msqrtloc,2,0),3,1),4,2),5,3)
 
  Mloc = np.linalg.inv(Msqrtloc)
  Msqrtloc = np.linalg.cholesky(np.linalg.inv(Msqrtloc))
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


def make_basis(var):
  sys.stdout.write('Constructing POD basis functions' + '\n')
  sys.stdout.write('Starting iteration = ' + str(start) + '\n')
  sys.stdout.write('End iteration = ' + str(end) + '\n')
  sys.stdout.write('Stride = ' + str(skip) + '\n')

  ## Create array for snapshots
  S = np.zeros((ndata,0) )


  for sol_folder in range(0,sol_end,sol_skip):
    for i in range(0,end,skip):
      sys.stdout.write('On sol folder ' + str(sol_folder) + ' and sol number ' + str(i).zfill(5) + '\r')
      sys.stdout.flush()
      loc_array = np.zeros(0)
      for j in range(0,n_blocks):
        check = 0
        sol_str = 'Solution_' + str(sol_folder) + '/npsol_block' + str(j) + '_'  + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          check = 1
          data = np.load(sol_str)['a']
          data = np.sum(Msqrt[j][None]*data[:,None,None,None,None],axis=(5,6,7,8) )
          loc_array = np.append(loc_array,data[var].flatten() )
      if (check == 1):
        S = np.append(S,loc_array[:,None],axis=1)

  U,Lam,Z = np.linalg.svd(S,full_matrices=False)
  ### Compute transformed basis
  N,K = np.shape(U)
  for i in range(0,K):
    tmp = np.reshape(U[:,i],np.shape(data[0]))
    tmp = np.sum(Msqrtinv[0]*tmp[None,None,None,None],axis=(4,5,6,7) )
    U[:,i] = tmp.flatten()
  return U,Lam

def truncate_basis(Lam,U,tol):
  ## create truncated basis corresponding to 99%, 99.9%, and 99.99%
  print(np.shape(U))
  energy = np.sum(Lam)
  nvec = np.size(Lam)
  indx = 1
  check = 0
  while (check == 0):
    rel_energy = np.sum(Lam[0:indx] )
    if (rel_energy/(energy + 1e-10) >= tol or energy <= 1e-10 ):
      check = 1
      sys.stdout.write(str(tol*100) + '\% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
      return U[:,0:indx]
    else:
      indx += 1

V = []
Lam = []
for i in range(0,nvars):
  Vt,Lamt = make_basis(i)
  V.append(Vt)
  Lam.append(Lamt)


def truncate(V,Lam,tol):
  Vtrunc = []
  sz = []
  for i in range(0,nvars):
    Vt = truncate_basis(Lam[i],V[i],tol)
    Vtrunc.append(Vt)
    sz.append(np.shape(Vt) )
  
  nrows = nvars*ndata
  ncols = 0
  for i in range(0,nvars):
    ncols += sz[i][1]
  
  V = np.zeros((nrows,ncols) )
  start = 0
  for i in range(0,nvars):
    sz = np.shape(Vtrunc[i])
    end = start + sz[1]
    V[i*sz[0]:(i+1)*sz[0],start:end] = Vtrunc[i]
    start = end
  
  np.savez('pod_vector_' + str(tol),V=V)



truncate(V,Lam,0.99)
truncate(V,Lam,0.999)
truncate(V,Lam,0.9999)
