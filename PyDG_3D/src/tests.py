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

def vortexICS(x,y,qnum):
  gamma = 1.4
  y0 = 5.
  x0 = 5.
  Cv = 5/2.
  Cp = Cv*gamma
  R = Cp - Cv
  nx,ny = np.shape(x)
  q = np.zeros((4,nx,ny))
  T = np.zeros((nx,ny))
  rho = np.zeros((nx,ny))
  u = np.zeros((nx,ny))
  v = np.zeros((nx,ny))
  E = np.zeros((nx,ny))
  r = ( (x - x0)**2 + (y - y0)**2 )**0.5
  beta = 5.
  pi = np.pi
  T[:,:] = 1. - (gamma - 1.)*beta**2/(8.*gamma*pi**2)*np.exp(1. - r**2)
  rho[:,:] = T**(1./(gamma - 1.))
  u[:,:] = 0. + beta/(2.*pi)*np.exp( (1. - r**2)/2.)*-(y - y0)
  v[:,:] = 0. +  beta/(2.*pi)*np.exp( (1. - r**2)/2.)*(x - x0)
  E[:,:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*E
  p = (gamma - 1.)*q[0]*(q[3]/q[0] - 0.5*q[1]**2/q[0]**2 - 0.5*q[2]**2/q[0]**2)
  T = p/(rho*R)
  E[:,:] = Cv*T + 0.5*(u**2 + v**2)
  return q[qnum]



L = 10.#2.*np.pi
L = 10.#2.*np.pi
Nel = np.array([2**7,2**7])
order = 2
nu = 1.e-1
x = np.linspace(0,L,Nel[0]+1)
y = np.linspace(0,L,Nel[1]+1)

t = 0
dt = 1.e-3
et = 10.
iteration = 0
save_freq = 25
eqns = equations('Navier-Stokes')
main = variables(Nel,order,eqns,nu,x,y,t,et,dt,iteration,save_freq)
schemes = fschemes('rusanov','central')

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
