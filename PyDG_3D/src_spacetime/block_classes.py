import numpy as np
from mpi4py import MPI
from turb_models import *
from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs

class blockClass:
  def __init__(self,nblocks,starting_rank,procx,procy,procz,et,dt,save_freq,turb_str,Nel_block,order,eqns):
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    self.starting_rank = starting_rank
    ### Assign block number to each processor
    procnumber = 0
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

    self.a = np.zeros(solution_size)
    self.a0 = np.zeros(solution_size)
    self.RHS = np.zeros(solution_size)
    ### Check turbulence models
    self.turb_str = turb_str
    check = 0
    if (turb_str == 'tau-model'):
      self.getRHS_REGION_OUTER = tauModelLinearized
      check = 1
    if (turb_str == 'orthogonal subscale'):
      self.getRHS_REGION_OUTER = orthogonalSubscale
      check = 1
    if (turb_str == 'orthogonal subscale POD'):
      self.getRHS_REGION_OUTER = orthogonalSubscale_POD
      check = 1
    if (check == 0):
      self.getRHS_REGION_OUTER = DNS
      print('Error, ' + turb_str + ' not found. Using DNS')
    else:
      if (self.mpi_rank == 0):
         print('Using turb model ' + turb_str)

    def getRHS_REGION_INNER(self,eqns):
      for region in self.region:
        region.basis.reconstructU(region,region.a)
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS(self,eqns)
      for region in self.region:
        region.basis.applyMassMatrix(region,region.RHS)
  

    def getRHS_REGION_INNER_ELEMENT(self,eqns):
      for region in self.region:
        region.basis.reconstructU(region,region.a)
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for region in self.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      eqns.getRHS_element(self,eqns)
      for region in self.region:
        region.basis.applyMassMatrix(region,region.RHS)


    self.getRHS_REGION_INNER = getRHS_REGION_INNER 
    self.getRHS_REGION_INNER_ELEMENT = getRHS_REGION_INNER_ELEMENT 


