import numpy as np
from init_Classes import *
from DG_functions import reconstructU
from MPI_functions import gatherSol
import matplotlib.pyplot as plt
from timeSchemes import advanceSol
import time
def getIC(main,f,qnum):
  ## First perform integration in x
  for i in range(0,main.Npx):
    for j in range(0,main.Npy):
      xloc = (main.x[i] + main.x[i] + main.dx)/2. + main.zeta/2.*(main.dx)
      yloc = (main.y[j] + main.y[j] + main.dy)/2. + main.zeta/2.*(main.dy)
      for p in range(0,main.order):
        for q in range(0,main.order):
          tg = np.ones((main.order,main.order))
          main.a.a[qnum,p,q,i,j] = (2.*p+1)/2.*(2.*q+1)/2.*np.sum(main.weights[:,None]*main.weights[None,:]*main.w[p,:,None]*main.w[q,None,:]*f(xloc[:,None]*tg,yloc[None,:]*tg,qnum))



Nel = np.array([2**7,2**7])
order = 2
nu = 1.e-1
x = np.linspace(0,L,Nel[0]+1)
y = np.linspace(0,L,Nel[1]+1)
z = np.linspace(0,L,Nel[2]+1)

t = 0
dt = 1.e-3
et = 10.
iteration = 0
save_freq = 25
eqns = equations('Linear-Advection')
main = variables(Nel,order,eqns,nu,x,y,t,et,dt,iteration,save_freq)
schemes = fschemes('central','central')

for qnum in range(0,eqns.nvars):
  getIC(main,vortexICS,qnum)

#print(np.linalg.norm(main.u[0]))
t0 = time.time()
while (main.t <= main.et - main.dt/2):
  if (main.iteration%main.save_freq == 0):
    reconstructU(main,main.a)
    uG = gatherSol(main,eqns,main.a)
    if (main.mpi_rank == 0):
      print('t = ' + str(main.t),'rho norm = ' + str(np.linalg.norm(uG[0])) )
      plt.clf()
      plt.contourf(uG[0,0,0,:,:],100)
      plt.pause(0.0001)
  advanceSol(main,eqns,schemes)
print('Final Time = ' + str(time.time() - t0))
