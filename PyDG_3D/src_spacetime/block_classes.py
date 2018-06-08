import numpy as np
from mpi4py import MPI
from turb_models import *
from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs

class blockClass:
  def __init__(self,nblocks,starting_rank,procx,procy,procz,et,dt,save_freq,turb_str):
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

    ### Check turbulence models
    self.turb_str = turb_str
    check = 0
    if (turb_str == 'tau-model'):
      self.getRHS_REGION_OUTER = tauModelLinearized
      check = 1
    if (turb_str == 'orthogonal subscale'):
      self.getRHS_REGION_OUTER = orthogonalSubscale
      check = 1
    if (check == 0):
      self.getRHS_REGION_OUTER = DNS
    else:
      if (self.mpi_rank == 0):
         print('Using turb model ' + turb_str)

    def getRHS_REGION_INNER(self,eqns):
      for main in self.region:
        main.basis.reconstructU(main,main.a)
        main.a.uR[:],main.a.uL[:],main.a.uU[:],main.a.uD[:],main.a.uF[:],main.a.uB[:] = main.basis.reconstructEdgesGeneral(main.a.a,main)

      for main in self.region:
        main.a.uR_edge[:],main.a.uL_edge[:],main.a.uU_edge[:],main.a.uD_edge[:],main.a.uF_edge[:],main.a.uB_edge[:] = sendEdgesGeneralSlab(main.a.uL,main.a.uR,main.a.uD,main.a.uU,main.a.uB,main.a.uF,main,self)

      eqns.getRHS(self,eqns)
    
    self.getRHS_REGION_INNER = getRHS_REGION_INNER 

