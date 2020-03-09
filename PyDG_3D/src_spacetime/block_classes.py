import numpy as np
from mpi4py import MPI
from turb_models import *
from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs
#from sklearn.externals import joblib
import logging
#from regression_class import regression_class
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#try:
#  from keras.models import load_model
#except:
#  if (MPI.COMM_WORLD.Get_rank() == 0):
#    logger.warning("Keras not found, may have issues with ML models")



class blockClass:
  def __init__(self,nblocks,starting_rank,procx,procy,procz,et,dt,save_freq,turb_str,Nel_block,order,eqns,datatype='float'):
    self.datatype = datatype
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    self.starting_rank = starting_rank
    ### Assign block number to each processor
    procnumber = 0
    self.n_blocks = nblocks
    self.Nel_block = Nel_block
    self.nblocks = 0
    self.mpi_regions_owned = []
    self.procx = procx
    self.procy = procy
    self.procz = procz
    self.nprocs = np.zeros(nblocks)
    for i in range(0,nblocks):
      self.nprocs[i] = procx[i]*procy[i]*procz[i]
      procnumber_new = starting_rank[i] + procx[i]*procy[i]*procz[i]
      if (self.mpi_rank >= starting_rank[i] and self.mpi_rank < procnumber_new):
        self.mpi_regions_owned.append(i)
        self.nblocks += 1
        #print(self.mpi_rank,self.mpi_regions_owned,starting_rank[i]) 
   #self.nblocks = nblocks
    self.region_owned_local = range(0,np.size(self.mpi_regions_owned))
    self.region = []
    self.t = 0
    self.iteration = 0
    self.dt = dt
    self.et = et
    self.save_freq = save_freq


    ### Create global arrays across the entire region for the state vector and the RHS.
    solution_size = 0
    self.global_start_indx = np.zeros(nblocks,dtype='int') #start and end indices to put GLOBAL quantities into region-wise quantities
    self.global_end_indx = np.zeros(nblocks,dtype='int')
    self.global_end_indx[0] = np.prod(Nel_block[0])*np.prod(order)*eqns.nvars
    for i in range(1,nblocks):
      self.global_start_indx[i] = self.global_end_indx[i-1]
      self.global_end_indx[i] = self.global_start_indx[i] + np.prod(Nel_block[i])*np.prod(order)*eqns.nvars

    self.solution_end_indx = np.zeros(0,dtype='int')
    self.solution_start_indx = np.zeros(0,dtype='int')
    for i in self.mpi_regions_owned:
      self.solution_start_indx = np.append(self.solution_start_indx,solution_size)
      Npx = int(float(Nel_block[i][0] / procx[i]))
      Npy = int(float(Nel_block[i][1] / procy[i]))
      Npz = int(float(Nel_block[i][2] / procz[i]))
      Npt = Nel_block[i][3]
      solution_size += eqns.nvars*order[0]*order[1]*order[2]*order[3]*Npx*Npy*Npz*Npt
      self.solution_end_indx = np.append(self.solution_end_indx,solution_size)

    self.a = np.zeros(solution_size,dtype=datatype)
    self.a0 = np.zeros(solution_size,dtype=datatype)
    self.RHS = np.zeros(solution_size,dtype=datatype)
    ### Check turbulence models
    self.turb_str = turb_str
    check = 0
    if (turb_str == 'residualMinimization_POD'):
      self.getRHS_REGION_OUTER = residualMinimization_POD
      check = 1
    if (turb_str == 'ML_CUSTOM'):
      self.getRHS_REGION_OUTER = ML_CUSTOM
      V = np.load('pod_basis.npz')['V'] 
      output_size = np.shape(V)[1]
 
      regression_model = 'linear_model_scalar' 
      if (regression_model == 'linear_model'):
        input_shape = np.array([np.shape(V)[1],1])
        self.rho = np.zeros(np.shape(V)[1])
        try:
          coefs = np.load('coefs.npz')['coefs'] 
          self.regressor = regression_class(regression_model,output_size,input_shape,coefs)
          print('Using previously tuned coefficients')
        except:
          self.regressor = regression_class(regression_model,output_size,input_shape)

      if (regression_model == 'linear_model_scalar'):
        input_shape = np.array([np.shape(V)[1],1])
        self.rho = np.zeros(np.shape(V)[1])
        try:
          coefs = np.load('coefs.npz')['coefs'] 
          self.regressor = regression_class(regression_model,output_size,input_shape,coefs)
          print('Using previously tuned coefficients')
        except:
          self.regressor = regression_class(regression_model,output_size,input_shape)

      check = 1
    if (turb_str == 'adjoint'):
      self.getRHS_REGION_OUTER = adjoint 
      check = 1
    if (turb_str == 'ROM'):
      self.getRHS_REGION_OUTER = ROM 
      check = 1
    if (turb_str == 'QDEIM'):
      self.getRHS_REGION_OUTER = QDEIM
      check = 1
    if (turb_str == 'tau-model'):
      self.getRHS_REGION_OUTER = tauModelLinearized
      check = 1
    if (turb_str == 'orthogonal subscale'):
      self.getRHS_REGION_OUTER = orthogonalSubscale
      check = 1
#    if (turb_str == 'GP POD'):
#      self.getRHS_REGION_OUTER = GP_POD
#      self.model = joblib.load('gp_model.joblib')
#      self.gp_info = np.load('gp_info.npz')
#      check = 1
    if (turb_str == 'KERAS POD'):
      self.getRHS_REGION_OUTER = KERAS_POD
      self.model = load_model('keras_closure_model.h5')
      self.model_info = np.load('keras_info.npz')
      check = 1
    if (turb_str == 'orthogonal subscale COLLOCATE'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_COLLOCATE
      check = 1
    if (turb_str == 'orthogonal subscale QDEIM'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_QDEIM
      check = 1
    if (turb_str == 'orthogonal subscale POD'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD
      check = 1
    if (turb_str == 'orthogonal subscale POD stochastic'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD_stochastic
      check = 1
    if (turb_str == 'DNS stochastic'):
      self.getRHS_REGION_OUTER = DNS_stochastic
      check = 1
    if (turb_str == 't-model POD'):
      self.getRHS_REGION_OUTER = tmodel_POD
      check = 1
    if (turb_str == 'orthogonal subscale POD unsteady'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD_unsteady
      check = 1
    if (turb_str == 'orthogonal subscale POD unsteady stochastic'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD_unsteady_stochastic
      check = 1
    if (turb_str == 'LSTM POD'):
      from my_nn2 import build_model
      data = np.load('weights.npz')
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD_LSTM
      neurons = 1
      depth = 1
      input_size = 1
      output_size = 1
      model_type = 'NN'
      cell_states = 1
      Xin = np.zeros((1,1,input_size))
      model = build_model(model_type,neurons,depth,cell_states,input_size,output_size)
      self.model=model
      self.model.state_save = np.zeros((np.size(self.a),1))
      #self.model.state_save = self.model.state_save[:,None,None]
      self.model.update_weights(self.model,data['weights'])
      check = 1

    if (turb_str == 'RNN POD'):
      from my_nn import build_model
      data = np.load('weights.npz')
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD_RNN
      neurons = 1
      depth = 1
      input_size = 1
      output_size = 1
      model_type = 'RNN'
      Xin = np.zeros((1,1,input_size))
      model = build_model(model_type,neurons,depth,input_size,output_size)
      self.model=model
      self.model.state_save = np.zeros(np.size(self.a))
      self.model.state_save = self.model.state_save[:,None,None]
      self.model.update_weights(self.model,data['weights'])
      check = 1

    if (check == 0):
      self.getRHS_REGION_OUTER = DNS
      print('Error, ' + turb_str + ' not found. Using DNS')
    else:
      if (self.mpi_rank == 0):
         print('Using turb model ' + turb_str)


    def getRHS_REGION_INNER_QDEIM(self,eqns):
#      for region in self.region:
#        region.basis.reconstructU(region,region.a)
#        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        cell_ijk = region.cell_ijk
        #stencil_ijk = region.stencil_ijk
        stencil_ijk = region.viscous_stencil_ijk
        stencil2_ijk = region.stencil_ijk

        #region.a.u[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] =  reconstructUGeneral_einsum(region,region.a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
        #region.a.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] =  reconstructUGeneral_einsum(region,region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
        #region.a.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] =  reconstructUGeneral_tensordot(region,region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
        region.a.u_hyper_stencil =  reconstructUGeneral_tensordot(region,region.a.a[:,:,:,:,:,stencil2_ijk[5][0],stencil2_ijk[6][0],stencil2_ijk[7][0],stencil2_ijk[8][0]])
        region.a.u_hyper_cell =  reconstructUGeneral_tensordot(region,region.a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])

        #print('hi',np.linalg.norm(region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]))
        v1,v2,v3,v4,v5,v6 = region.basis.reconstructEdgesGeneral(region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region)
        region.a.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v1[:]
        region.a.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v2[:]
        region.a.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v3[:]
        region.a.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v4[:]
        region.a.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v5[:]
        region.a.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v6[:]
        stencil_ijk = region.stencil_ijk

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS_hyper(self,eqns)
      for region in self.region:
        #region.RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = np.einsum('ijklpqrs...,zpqrs...->zijkl...',region.Minv[:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]], region.RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
        #print(np.shape(region.Minv[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]))
        region.RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = np.sum(region.Minv[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*region.RHS[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],axis=(5,6,7,8) )
        #region.basis.applyMassMatrix(region,region.RHS)


    def getRHS_REGION_INNER(self,eqns):
      for region in self.region:
        region.basis.reconstructU(region,region.a)
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS(self,eqns)
      for region in self.region:
        region.basis.applyMassMatrix(region,region.RHS)

    def getRHS_REGION_INNER_ROM(self,eqns):
      for region in self.region:
        region.basis.reconstructU(region,region.a)
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS(self,eqns)
      #Note we don't apply the mass matrix here as  the POD  basis is orthogonal
  

    def getRHS_REGION_INNER_ELEMENT(self,eqns):
      for region in self.region:
        region.basis.reconstructU(region,region.a)
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS_element(self,eqns)
      for region in self.region:
        region.basis.applyMassMatrix(region,region.RHS)

    #if (turb_str == 'QDEIM'):
    self.getRHS_REGION_INNER_QDEIM = getRHS_REGION_INNER_QDEIM 
    self.getRHS_REGION_INNER = getRHS_REGION_INNER 
    self.getRHS_REGION_INNER_ELEMENT = getRHS_REGION_INNER_ELEMENT 
    #if (turb_str == 'ROM'):
    self.getRHS_REGION_INNER_ROM = getRHS_REGION_INNER_ROM 



