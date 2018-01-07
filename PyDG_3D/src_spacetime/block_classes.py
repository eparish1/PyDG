import numpy as np
from mpi4py import MPI

class blockClass:
  def __init__(self,nblocks,starting_rank,procx,procy,et,dt,save_freq):
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
    self.nprocs = np.zeros(nblocks)
    for i in range(0,nblocks):
      self.nprocs[i] = procx[i]*procy[i]
      procnumber_new = starting_rank[i] + procx[i]*procy[i]
      if (self.mpi_rank >= starting_rank[i] and self.mpi_rank < procnumber_new):
        self.mpi_regions_owned.append(i)
        self.nblocks += 1
        print(self.mpi_rank,self.mpi_regions_owned,starting_rank[i]) 
   #self.nblocks = nblocks
    self.region = []
    self.t = 0
    self.iteration = 0
    self.dt = dt
    self.et = et
    self.save_freq = save_freq
    def getRHS_REGION(self,eqns):
      nblocks = np.size(self.nblocks)
      for i in range(0,self.nblocks):
        self.region[i].getRHS(self.region[i],self.region[i],eqns,args=None)
    self.getRHS_REGION = getRHS_REGION    

