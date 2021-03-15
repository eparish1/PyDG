import numpy as np
from mpi4py import MPI
from timeSchemes import *
from rom_time_schemes import *
from linear_solvers import *
from nonlinear_solvers import *
import pickle
#from shallow_autoencoder import * 
import torch

class timeschemes:
  def init_pod(time_scheme,regionManager):
    V = np.load('pod_basis.npz')['V']
    #data_R = np.load('rhs_pod_basis.npz')
    n_basis = np.shape(V)[1]
    regionManager.V = np.zeros((0,n_basis) )
    regionManager.a_pod = np.zeros(n_basis)
    for region in regionManager.region:
      start_indx = regionManager.global_start_indx[region.region_number]
      end_indx = regionManager.global_end_indx[region.region_number]
      V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
      for i in range(0,n_basis):
        V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[start_indx:end_indx,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
      regionManager.V = np.append(regionManager.V,np.reshape(V2,(np.size(region.a.a),n_basis) ) , axis=0) 

  def init_manifold(time_scheme,regionManager):
    region = regionManager.region[0]
    N = region.Nglobal_stencil_cells
    depth = 3
    regionManager.manifold_model = ShallowAutoencoder(1,N,10,depth)
    regionManager.manifold_model.load_state_dict(torch.load('manifold_model',map_location='cpu'))

  def __init__(self,regionManager,time_str='ExplicitRK4',lsolver_str='GMRes',nlsolver_str='Newton'):
    comm = MPI.COMM_WORLD
    check_t = 0
    if (comm.Get_rank() == 0): print('Time marching is set to ' + time_str)
    if (time_str == 'backwardEuler_DLS'):
      check_t = 0
      self.advanceSol = backwardEuler_DLS
      self.args = None
    if (time_str == 'ExplicitRK4'):
      check_t = 0
      self.advanceSol = ExplicitRK4
      self.args = None
    if (time_str == 'ExplicitRK2'):
      check_t = 0
      self.advanceSol = ExplicitRK2
      self.args = None
    if (time_str == 'SSP_RK3'):
      check_t = 0
      self.advanceSol = SSP_RK3
      self.args = None
    if (time_str == 'SSP_RK3_Entropy'):
      check_t = 0
      self.advanceSol = SSP_RK3_Entropy
      self.args = None
    if (time_str == 'SSP_RK3_DOUBLEFLUX'):
      check_t = 0
      self.advanceSol = SSP_RK3_DOUBLEFLUX
      self.args = None
    if (time_str == 'crankNicolsonRom'):
      check_t = 0
      self.advanceSol = crankNicolsonRom
      self.args = ['GaussNewton']
      self.init_pod(regionManager)

    if (time_str == 'crankNicolsonManifoldRomCollocation'):
      check_t = 0
      self.advanceSol = crankNicolsonManifoldRomCollocation
      self.args = ['GaussNewton']
      self.init_manifold(regionManager)
      N = np.size(regionManager.region[0].cell_list)
      regionManager.numStepsInWindow = 2
      K = 10
      J_sparsity = np.zeros((N*regionManager.numStepsInWindow,K*regionManager.numStepsInWindow))
      J_sparsity[0:N,0:K] = 1
      for i in range(1,regionManager.numStepsInWindow):
        J_sparsity[N*i:N*(i+1),K*(i-1):K*(i+1)] = 1
      regionManager.J_sparsity = scipy.sparse.csr_matrix(J_sparsity)

    if (time_str == 'crankNicolsonRomCollocation'):
      check_t = 0
      self.advanceSol = crankNicolsonRomCollocation
      self.args = ['GaussNewton']
      self.init_pod(regionManager)
    if (time_str == 'SSP_RK3_POD_QDEIM_VALIDATE'):
      check_t = 0
      self.advanceSol = SSP_RK3_POD_QDEIM_VALIDATE
      self.args = None
      self.init_pod(regionManager)
    if (time_str == 'SSP_RK3_POD_COLLOCATE'):
      check_t = 0
      self.advanceSol = SSP_RK3_POD_COLLOCATE
      self.args = None
      self.init_pod(regionManager)

    if (time_str == 'SSP_RK3_POD_QDEIM'):
      check_t = 0
      self.advanceSol = SSP_RK3_POD_QDEIM
      self.args = None
      self.init_pod(regionManager)

    if (time_str == 'SSP_RK3_POD_unsteady'):
      check_t = 0
      self.advanceSol = SSP_RK3_POD_unsteady
      self.args = None
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.V = np.zeros((0,n_basis) )
      regionManager.RHS2 = np.zeros(np.shape(regionManager.RHS))
      regionManager.K = np.zeros(np.shape(regionManager.a))
      for region in regionManager.region:
        start_indx = regionManager.global_start_indx[region.region_number]
        end_indx = regionManager.global_end_indx[region.region_number]
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[start_indx:end_indx,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.append(regionManager.V,np.reshape(V2,(np.size(region.a.a),n_basis) ) , axis=0) 
    if (time_str == 'backwardEuler_LSPG'):
      check_t = 0
      self.advanceSol = backwardEuler_LSPG
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 


    if (time_str == 'backwardEuler_LSPG_collocation_validate'):
      check_t = 0
      self.advanceSol = backwardEuler_LSPG_collocation_validate
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 


     
    if (time_str == 'SSP_RK3_POD'):
      check_t = 0
      self.advanceSol = SSP_RK3_POD
      self.args = None
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.V = np.zeros((0,n_basis) )
      for region in regionManager.region:
        start_indx = regionManager.global_start_indx[region.region_number]
        end_indx = regionManager.global_end_indx[region.region_number]
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[start_indx:end_indx,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 




    if (time_str == 'crankNicolson_GNAT'):
      check_t = 0
      self.advanceSol = crankNicolson_GNAT
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      data_R = np.load('rhs_pod_basis.npz')
      n_basis = np.shape(V)[1]
      regionManager.V = np.zeros((0,n_basis) )
      regionManager.MR = data_R['M']
      regionManager.VR = np.dot(V.transpose(),data_R['M'])
      regionManager.ZW_pinv = np.dot(data_R['V'].transpose(),data_R['M']) ## follow notation from carlberg implementation
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 


    if (time_str == 'crankNicolson_LSPG_QDEIM'):
      check_t = 0
      self.advanceSol = crankNicolson_LSPG_QDEIM
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      data_R = np.load('rhs_pod_basis.npz')
      n_basis = np.shape(V)[1]
      regionManager.V = np.zeros((0,n_basis) )
      regionManager.MR = data_R['M']
      regionManager.VR = np.dot(V.transpose(),data_R['M'])

      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 

    if (time_str == 'backwardEuler_LSPG_QDEIM'):
      check_t = 0
      self.advanceSol = backwardEuler_LSPG_QDEIM
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 
    if (time_str == 'backwardEuler_Galerkin_QDEIM'):
      check_t = 0
      self.advanceSol = backwardEuler_Galerkin_QDEIM
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 


    if (time_str == 'backwardEuler_LSPG_windowed'):
      check_t = 0
      self.advanceSol = backwardEuler_LSPG_windowed
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('st_pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 

    if (time_str == 'crankNicolson_LSPG_windowed'):
      check_t = 0
      self.advanceSol = crankNicolson_LSPG_windowed
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      with open ('wst_spatial', 'rb') as fp:
        regionManager.VsG = pickle.load(fp)
      with open ('wst_temporal', 'rb') as fp:
        regionManager.VtG = pickle.load(fp)
      with open ('gappy_wst_spatial', 'rb') as fp:
        regionManager.PsG = pickle.load(fp)[0]
      with open ('gappy_wst_temporal', 'rb') as fp:
        regionManager.PtG = pickle.load(fp)[0]


      regionManager.window_counter = 0

    if (time_str == 'crankNicolson_LSPG'):
      check_t = 0
      self.advanceSol = crankNicolson_LSPG
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 

    if (time_str == 'backwardEuler_POD'):
      check_t = 0
      self.advanceSol = backwardEuler_POD
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 



    if (time_str == 'crankNicolson_POD'):
      check_t = 0
      self.advanceSol = crankNicolson_POD
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
      V = np.load('pod_basis.npz')['V']
      n_basis = np.shape(V)[1]
      regionManager.a_pod = np.zeros(n_basis)
      for region in regionManager.region:
        V2 = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt,n_basis))
        for i in range(0,n_basis):
          V2[:,:,:,:,:,:,:,:,:,i] = np.reshape(V[:,i],(region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Nel[0],region.Nel[1],region.Nel[2],region.Nel[3]))[:,:,:,:,:,region.sx,region.sy,region.sz,:] 
        regionManager.V = np.reshape(V2,(np.size(region.a.a),n_basis) ) 


    if (time_str == 'SpaceTime'):
      check_t = 0
      self.advanceSol = spaceTime 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'SpaceTimeExperimental'):
      check_t = 0
      self.advanceSol = spaceTimeExperimental 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SpaceTimeSplitting'):
      check_t = 0
      self.advanceSol = spaceTimeSplitting 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
    if (time_str == 'SpaceTimePC'):
      check_t = 0
      self.advanceSol = spaceTimePC
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'fractionalStep'):
      check_t = 0
      self.advanceSol = fractionalStep
      self.linear_solver = linearSolver(lsolver_str)
      self.pressure_linear_solver = linearSolver('BICGSTAB')
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature,self.pressure_linear_solver]


    if (time_str == 'SpaceTimeIncomp'):
      check_t = 0
      self.advanceSol = spaceTimeIncomp 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SteadyState'):
      check_t = 0
      self.advanceSol = SteadyState
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'SteadyStateExperimental'):
      check_t = 0
      self.advanceSol = SteadyStateExperimental
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'BackwardEuler'):
      check_t = 0
      self.advanceSol = backwardEuler
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolson'):
      check_t = 0
      self.advanceSol = CrankNicolson
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolsonEntropy'):
      check_t = 0
      self.advanceSol = CrankNicolsonEntropy
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolsonEntropyMZ'):
      check_t = 0
      self.advanceSol = CrankNicolsonEntropyMZ
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'CrankNicolsonIncomp'):
      check_t = 0
      self.advanceSol = CrankNicolsonIncomp
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'StrangSplitting'):
      check_t = 0
      self.advanceSol = StrangSplitting
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = True
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SDIRK2'):
      check_t = 0
      self.advanceSol = SDIRK2 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = True 
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
    if (time_str == 'SDIRK4'):
      check_t = 0
      self.advanceSol = SDIRK4 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False 
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

class nonlinearSolver:
  def __init__(self,SolverType='Newton',rtol=1e-8,printnorm=1):
    comm = MPI.COMM_WORLD
    if (comm.Get_rank() == 0): print('NL solver set to ' + SolverType)
    if (SolverType == 'Newton'):
      self.solve = newtonSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'NEJ'):
      self.solve = NEJSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'Newton_MG'):
      self.solve = newtonSolver_MG
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'Newton_PC'):
      self.solve = newtonSolver_PC2
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'pseudoTime'):
      self.solve = pseudoTimeSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'pseudoTime_MG'):
      self.solve = pseudoTimeSolver_MG
      self.rtol=rtol
      self.printnorm = printnorm


class linearSolver:
  def __init__(self,SolverType='GMRes',tol=1e-8,maxiter_outer=1,maxiter=15,printnorm=0):
    comm = MPI.COMM_WORLD
    if (SolverType == 'GMRes'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = GMRes
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'Jacobi'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = Jacobi
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'ADI'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = ADI
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'fGMRes'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = fGMRes
      self.solvePC = GMRes 
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'RungeKutta'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = rungeKutta
      self.solvePC = rungeKutta
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
    if (SolverType == 'BICGSTAB'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = BICGSTAB
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = 50
      self.printnorm = printnorm

