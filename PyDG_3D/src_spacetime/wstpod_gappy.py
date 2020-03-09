import numpy as np
import sys
import os
import pickle
import scipy
import scipy.linalg

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
  data = np.load(sol_str)['RHS']
  ndata += np.size(data)

ashp = np.shape(np.load(sol_str)['a'])
## Create array for snapshots
S = np.zeros((ndata,0) )

## get number of windows
try:
  window_size = int( sys.argv[1] )
except:
  window_size = 1

print('Window size =  ' + str(window_size))
# get starting iteration number
try:
  start = int( sys.argv[2] )
except:
  start = 0



# get final iteration number
try:
  end = int( sys.argv[3] )
except:
  end = 20000

#get strides
try:
  skip = int(sys.argv[4] )
except:
  skip = 1


try:
  sol_start = int(sys.argv[5] )
except:
  sol_start = 0


try:
  sol_end = int(sys.argv[6] )
except:
  sol_end = 1

try:
  sol_skip = int(sys.argv[7] )
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
#S2 = np.reshape(S,(ndata,nt,nparams) )
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

  Qs,Rs,Ps = scipy.linalg.qr(Us[:,0:indx_s].transpose(),pivoting=True )
  Ps = augmentP(Ps[0:indx_s])
  nsamples = np.size(Ps)
  P2s = np.zeros((ndata,nsamples))
  for i in range(0,nsamples):
    P2s[Ps[i],i] = 1
  ## Now make projection matrix
  Ms = gappy_leastsquares(Us,P2s)
  index_tuple = np.unravel_index(Ps,np.shape(data),order='C') ## generate tuple of indices for P
  #np.savez('rhs_pod_basis_' + str(percent),V=U[:,0:indx],index_global=P,index_tuple=index_tuple,M=M)#,testP=testP)

  ## Determine number of temporal basis
  check = 0
  while (check == 0):
    rel_energy_t = np.sum(Lamt[0:indx_t] )
    if (rel_energy_t/(energy_t + 1e-10) >= tol_t or indx_t >= window_size):
      check = 1
    else:
      indx_t += 1
  ## Create tensor product basis
  sys.stdout.write(str(tol_s) + ' of spatial energy = ' + str(indx_s) + '/' + str(nvec_s) + ' spatial basis vectors \n')
  sys.stdout.write(str(tol_t) + ' of temporal energy = ' + str(indx_t) + '/' + str(nvec_t) + ' temporal basis vectors \n')
  #np.savez('st_pod_basis_' + str(tol_s) + '_' + str(tol_t),Vs=Us[:,0:indx_s],Vt=Ut[:,0:indx_t])
  Qt,Rt,Pt = scipy.linalg.qr(Ut[:,0:indx_t].transpose(),pivoting=True )
  Pt = Pt[0:indx_t]# augmentP(Pt[0:indx_t])
  nsamples = np.size(Pt)
  P2t = np.zeros((window_size,nsamples))
  for i in range(0,nsamples):
    P2t[Pt[i],i] = 1
  ## Now make projection matrix
  print(Pt)
  Mt = gappy_leastsquares(Ut,P2t)
  index_tuple = np.unravel_index(Pt,np.shape(data),order='C') ## generate tuple of indices for P
  return Us[:,0:indx_s],Ut[:,0:indx_t],Ps,Pt

def gappy_leastsquares(Ut,P2):
  ## solve least squares problem
  tmp = np.linalg.pinv( np.dot(P2.transpose(),Ut))
  # now create "lifted" matrix
  M = np.dot(Ut,tmp)
  return M

def augmentP(cell_list):
  cell_ijk = np.unravel_index(cell_list, ashp)
  cell_ijk0 = np.copy(cell_ijk)
  for i in range(0,ashp[0]):
    tmp = np.copy(cell_ijk0)
    tmp[0][:] = i
    cell_ijk = np.append(cell_ijk,tmp,axis=1)

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


S_spatial = np.reshape(S,(ndata,nt*nparams) )
S_temporal = np.rollaxis(S,1 )
UsG = []
UtG = []
PsG = []
PtG = []
#Ntw = nt/nwindows
nwindows = nt/window_size#int(np.floor(nt*1./window_size))
print('Total of ' + str(nwindows) + ' windows ')
for window in range(0,nwindows):
  print(window)
  ### Start by making spatial basis
  Us,Lams,Z = np.linalg.svd(S_spatial[:,window*window_size:(window+1)*window_size],full_matrices=False) # Ns x Ks
  #Us,Lams,Z = np.linalg.svd(S_spatial[:,:],full_matrices=False) # Ns x Ks

  ### Now make temporal basis
  print(np.shape(S_temporal[window*window_size:(window+1)*window_size,:] ) )
  Ut,Lamt,Z = np.linalg.svd(S_temporal[window*window_size:(window+1)*window_size,:],full_matrices=False) # Nt x Kt

  Us_tmp,Ut_tmp,Ps_tmp,Pt_tmp = construct_st_basis(Us,Ut,Lams,Lamt,0.99,0.99)
  #Ut_tmp = np.eye(window_size)
  UsG.append(Us_tmp)
  UtG.append(Ut_tmp)
  PsG.append(Ps_tmp)
  PtG.append(Pt_tmp)


with open('gappy_wst_spatial', 'wb') as fp:
    pickle.dump([PsG,UsG], fp)

with open('gappy_wst_temporal', 'wb') as fp:
    pickle.dump([PtG,UtG], fp)
#np.savez('wst_pod_basis',Vs=UsG,Vt=UtG)










