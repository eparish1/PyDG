import numpy as np
import sys
import os
import scipy
import scipy.linalg
### script to make POD basis vectors
## Uses the QR decomposition for index selection. After picking indices, it adds
## all conserved variables at those select indices. Reuced matrices use gappy POD
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
  data = np.load(sol_str)['RHS'] ## Only generate this for the elements of a single conserved variable
  ndata += np.size(data)
ashp = np.shape(np.load(sol_str)['a'])
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


sys.stdout.write('Constructing POD basis functions')
sys.stdout.write('Starting iteration = ' + str(start) + '\n')
sys.stdout.write('End iteration = ' + str(end) + '\n')
sys.stdout.write('Stride = ' + str(skip) + '\n')


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
        #print('found ' + sol_str)
        data = np.load(sol_str)['RHS']
        loc_array = np.append(loc_array,data.flatten() )
    if (check == 1):
      S = np.append(S,loc_array[:,None],axis=1)



U,Lam,Z = np.linalg.svd(S,full_matrices=False)

#np.savez('rhs_pod_basis_full',V=U,Lam=Lam)

## create truncated basis corresponding to 99%, 99.9%, and 99.99%

energy = np.sum(Lam)
nvec = np.size(Lam)

##
def gappy_leastsquares(Ut,P2):
  ## solve least squares problem
  ## looks like this:
  #tmp = np.linalg.inv( np.dot(Ut.transpose() , np.dot(P2, np.dot(P2.transpose(),Ut))))
  #tmp = np.dot(tmp,Ut.transpose())
  #tmp = np.dot(tmp,P2)
  ## but we can just use pinv, should be better
  tmp = np.linalg.pinv( np.dot(P2.transpose(),Ut)) 
  # now create "lifted" matrix
  M = np.dot(Ut,tmp)
  return M
## create a function that adds all conserved variables to the stencil
def augmentP(cell_list):
  cell_ijk = np.unravel_index(cell_list, ashp)
  print(np.shape(cell_ijk))
  cell_ijk0 = np.copy(cell_ijk)
  for i in range(0,ashp[0]):
    tmp = np.copy(cell_ijk0)
    tmp[0][:] = i
    cell_ijk = np.append(cell_ijk,tmp,axis=1)

  ## now add all of the boundary points (2D for now)
#  cell_ijk0 = np.copy(cell_ijk)
#  for j in range(0,ashp[6]):
#    for k in range(0,ashp[7]):
#      tmp = np.copy(cell_ijk0)
#      tmp[5][:] = 0
#      tmp[6][:] = j
#      tmp[7][:] = k
#      cell_ijk = np.append(cell_ijk,tmp,axis=1)
#      tmp = np.copy(cell_ijk0)
#      tmp[5][:] = ashp[5] - 1
#      tmp[6][:] = j
#      tmp[7][:] = k
#      cell_ijk = np.append(cell_ijk,tmp,axis=1)
#
#  for i in range(0,ashp[5]):
#    for k in range(0,ashp[7]):
#      tmp = np.copy(cell_ijk0)
#      tmp[5][:] = i
#      tmp[6][:] = 0
#      tmp[7][:] = k
#      cell_ijk = np.append(cell_ijk,tmp,axis=1)
#      tmp = np.copy(cell_ijk0)
#      tmp[5][:] = i
#      tmp[6][:] = ashp[6] - 1 
#      tmp[7][:] = k
#      cell_ijk = np.append(cell_ijk,tmp,axis=1)
  cell_ijk0 = np.copy(cell_ijk)
  for i in range(0,ashp[1]):
    for j in range(0,ashp[2]):
      for k in range(0,ashp[3]):
        for l in range(0,ashp[4]): 
          tmp = np.copy(cell_ijk0)
          tmp[1][:] = i
          tmp[2][:] = j
          tmp[3][:] = k
          tmp[4][:] = l
          cell_ijk = np.append(cell_ijk,tmp,axis=1)
  cell_list = np.ravel_multi_index( (cell_ijk[0,:],cell_ijk[1,:],cell_ijk[2,:],cell_ijk[3,:],cell_ijk[4,:],cell_ijk[5,:],cell_ijk[6,:],cell_ijk[7,:],cell_ijk[8,:]),ashp)
  cell_list = np.unique(cell_list)
  return cell_list 

def write_output(percent):
  indx = 1
  check = 0
  while (check == 0):
    rel_energy = np.sum(Lam[0:indx] )
    if (rel_energy/energy >= percent*1./100. ):
      print(indx)
      check = 1
      ## COMPUTE QR
      Q,R,P = scipy.linalg.qr(U[:,0:indx].transpose(),pivoting=True )
      P = augmentP(P[0:indx])
      basis_size = int( np.size(P)/4.) 
      
      nsamples = np.size(P)
      n = np.shape(U)[0]
      Ut = U[:,0:basis_size]
      P2 = np.zeros((n,nsamples))
      for i in range(0,nsamples):
        #print(indx)
        P2[P[i],i] = 1
      ## Now make projection matrix
      M = gappy_leastsquares(Ut,P2)
      index_tuple = np.unravel_index(P,np.shape(data),order='C') ## generate tuple of indices for P
      np.savez('rhs_pod_basis_' + str(percent),V=U[:,0:basis_size],index_global=P,index_tuple=index_tuple,M=M)#,testP=testP)
      sys.stdout.write(str(percent) + '% of energy = ' + str(indx) + '/' + str(nvec) + ' basis vectors \n')
    else:
      indx += 1

write_output(90) #95%
write_output(95) #95%
write_output(97) #97#
write_output(99) #99%
#write_output(99.9) #99.9#
