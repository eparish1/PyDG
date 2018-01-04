import numpy as np

class blockClass:
  def __init__(self,nblocks,et,dt,save_freq):
    self.nblocks = nblocks
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

