import numpy as np

class gasClass:
  def __init__(self,gamma=1.4,Cv=5./2.):
    self.gamma = gamma
    self.Cv = Cv
    self.Cp = Cv*gamma
    self.R = self.Cp - self.Cv
