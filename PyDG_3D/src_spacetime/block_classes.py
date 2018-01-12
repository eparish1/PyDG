import numpy as np
from mpi4py import MPI
from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs

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
    self.region_owned_local = range(0,np.size(self.mpi_regions_owned))
    self.region = []
    self.t = 0
    self.iteration = 0
    self.dt = dt
    self.et = et
    self.save_freq = save_freq
    def getRHS_REGION(self,eqns):
      nblocks = np.size(self.nblocks)
   
      for i in self.regions_owned_local:
        region = self.region[i]
        region.basis.reconstructU(region,region.a)

      for i in self.regions_owned_local:
        region = self.region[i]
        region.a.uR[:],region.a.uL[:],region.a.uU[:],region.a.uD[:],region.a.uF[:],region.a.uB[:] = region.basis.reconstructEdgesGeneral(region.a.a,region)

      for i in self.regions_owned_local:
        region = self.region[i]
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,self)

      for i in self.regions_owned_local:
        self.region[i].getRHS(self,self.region[i],self.region[i],eqns)
        self.region[i].basis.applyMassMatrix(self.region[i],self.region[i].RHS)

      for i in self.regions_owned_local:
        addInviscidFlux(self.region[i],self.region[i],eqns,args,args_phys)
        eqns.evalFluxXYZ(self.region[i],self.region[i].a.u,self.region[i].iFlux.fx,self.region[i].iFlux.fy,self.region[i].iFlux.fz,args_phys)

        addSecondaryViscousContribution_BR1(self.region[i],self.region[i],eqns)
    self.getRHS_REGION = getRHS_REGION    

