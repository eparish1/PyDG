import numpy as np
import sys
import os
from mpi4py import MPI
import copy
comm = MPI.COMM_WORLD
from MPI_functions import *
mpi_rank = comm.Get_rank()
num_processes = comm.Get_size()
starting_rank = 0
print(num_processes)
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


ashp = np.shape(data)
nvars = ashp[0]
order = ashp[1:5]
Nel = ashp[5::]
Npx = int(float(Nel[0] / procx))
Npy = int(float(Nel[1] / procy)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
Npz = int(float(Nel[2] / procz))
Npt = Nel[-1]

def gatherBasisVector(a):
  if (mpi_rank == 0):
    aG = np.zeros((nvars,order[0],order[1],order[2],order[3],Nel[0],Nel[1],Nel[2],Nel[3]))
    aG[:,:,:,:,:,0:Npx,0:Npy,:] = a[:]
    for i in range(1,num_processes):
      loc_rank = i - starting_rank
      data = np.zeros(np.shape(a)).flatten()
      comm.Recv(data,source=loc_rank + starting_rank,tag = loc_rank + starting_rank)

      xL = (int(loc_rank)%int(procx*procy))%int(procx)*Npx
      xR = (int(loc_rank)%int(procx*procy))%int(procx)*Npx + Npx
      yD = (int(loc_rank)%int(procx*procy))//int(procx)*Npy
      yU = (int(loc_rank)%int(procx*procy))//int(procx)*Npy + Npy
      zB = (int(loc_rank)//int(procx*procy))*Npz
      zF = (int(loc_rank)//int(procx*procy))*Npz + Npz
      
      aG[:,:,:,:,:,xL:xR,yD:yU,zB:zF] = np.reshape(data,(nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npz,Npt))
    return aG.flatten()
  else:
    comm.Send(a.flatten(),dest=starting_rank,tag=mpi_rank)


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
  Nel = np.shape(grid['Minv'])[8::]
  Npx = int(float(Nel[0] / procx))
  Npy = int(float(Nel[1] / procy)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
  Npz = int(float(Nel[2] / procz))
  Npt = Nel[-1]
  
  sx = slice(int(((int(mpi_rank - starting_rank) % int(procx *procy)) % int(procx))     *Npx), \
                  int(((int(mpi_rank - starting_rank) % int(procx * procy)) % int(procx) + 1) *Npx))
   
  sy = slice(int(((int(mpi_rank - starting_rank) % int(procx * procy)) // int(procx))     *Npy), \
                  int(((int(mpi_rank - starting_rank) % int(procx * procy)) // int(procx) + 1) *Npy))
  
  sz = slice(int(((int(mpi_rank - starting_rank) // int(procx * procy)))     *Npz), \
                int(((int(mpi_rank - starting_rank) // int(procx * procy)) + 1) *Npz))



  Msqrtloc = grid['Minv'][:,:,:,:,:,:,:,:,sx,sy,sz]
  shp = np.shape(Msqrtloc)
  ndofs = np.prod(np.shape(Msqrtloc)[0:4])
  Msqrtloc = np.reshape(Msqrtloc, (ndofs,ndofs,Npx,Npy,Npz,Npt) )
  Msqrtloc = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis(Msqrtloc,2,0),3,1),4,2),5,3)
 
  Mloc = np.linalg.inv(Msqrtloc)

  num_axis = np.size(np.shape(Msqrtloc))
  axis_no = np.array(range(0,num_axis),dtype='int')
  axis_no[-1] = axis_no[-1] - 1
  axis_no[-2] = axis_no[-2] + 1
  Msqrtloc = np.linalg.cholesky(Mloc)
  Msqrtloc = np.transpose(Msqrtloc,axis_no)

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
      data = np.load(sol_str)['a'][:,:,:,:,:,sx,sy,sz]
      ashp_mpi = np.shape(data)
      data = np.sum(Msqrt[j][None]*data[:,None,None,None,None],axis=(5,6,7,8) )
      loc_array = np.append(loc_array,data.flatten() )
  if (check == 1):
    if (i == 0):
      S = loc_array[:,None]
    else:
      S = np.append(S,loc_array[:,None],axis=1)

if (mpi_rank == 0):
  print('Computing A^T A')
Kern = A_transpose_dot_b(S,S)
if (mpi_rank == 0):
  print('Computing SVD, matrix size = ' , np.shape(Kern))
u,sigma_sqr,_ = np.linalg.svd(Kern)
if (mpi_rank == 0):
  print('Done with SVD!')
sigma = np.sqrt(sigma_sqr) 
U = np.empty(np.shape(S) )
if (mpi_rank == 0):
  print('Computing basis vectors, U = 1/sigma S*U')
U[:] = np.dot(S, 1./sigma * u)
if (mpi_rank == 0):
  print('Done!')
Lam = sigma 

N,K = np.shape(U)
for i in range(0,K):
  tmp = np.reshape(U[:,i],np.shape(data))
  tmp = np.sum(Msqrtinv[0][None]*tmp[:,None,None,None,None],axis=(5,6,7,8) )
  U[:,i] = tmp.flatten()

'''
Ortho = np.zeros((K,K))
for i in range(0,K):
  for j in range(0,K):
    tmp = np.reshape(U[:,i],np.shape(data))
    tmp = np.sum(M[0][None]*tmp[:,None,None,None,None],axis=(5,6,7,8) )
    tmp = tmp.flatten()
    tmp2 = U[:,j]
    Ortho[i][j] = np.dot(tmp,tmp2)
print(Ortho)
'''   

## POD bases are distributed across ranks, 
#U2,Lam2,Z2 = np.linalg.svd(S,full_matrices=False)
#
#### Compute t
#N,K = np.shape(U)
#for i in range(0,K):
#  tmp = np.reshape(U2[:,i],np.shape(data))
#  tmp = np.sum(Msqrtinv[0][None]*tmp[:,None,None,None,None],axis=(5,6,7,8) )
#  #U2[:,i] = tmp.flatten()
#
#print(np.shape(U2),np.shape(U))
#print('Difference = ' + str(np.linalg.norm(np.abs(U2) - np.abs(U))))

def truncateBasis(U,Lam,tol):
  energy = np.sum(Lam**2)
  cum_energy = np.cumsum(Lam**2)
  rel_energy = cum_energy / energy
  K = np.size(rel_energy[rel_energy <= tol] )
  UG = gatherBasisVector(np.reshape(U[:,0],ashp_mpi))
  if (mpi_rank == 0):
    UG = UG[:,None]
  for i in range(1,K):
    UGt = gatherBasisVector(np.reshape(U[:,i],ashp_mpi))
    if (mpi_rank == 0):
      UG = np.append(UG,UGt[:,None],axis=1)
  if (mpi_rank == 0): 
    np.savez('pod_basis_' + str(tol) ,V=UG)
    sys.stdout.write(str(100*tol) + '% of energy = ' + str(np.shape(UG)[1]) + '/' + str(np.size(Lam)) + ' basis vectors \n')
  return UG
for tol in tol_a:
  Utmp = truncateBasis(U,Lam,tol)

