import numpy as np
import os
import sys
sys.path.append("../../src")

from init_Classes import *
from DG_functions import reconstructU
from MPI_functions import gatherSolSlab,gatherSolSpectral
import matplotlib.pyplot as plt
from timeSchemes import advanceSol
import time
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


def getGlobGrid(x,y,z,zeta):
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  dz = z[1] - z[0]
  Nelx,Nely,Nelz = np.size(x),np.size(y),np.size(z)
  order = np.size(zeta)
  xG = np.zeros(((np.size(x)-1)*np.size(zeta)))
  yG = np.zeros(((np.size(y)-1)*np.size(zeta)))
  zG = np.zeros(((np.size(z)-1)*np.size(zeta)))
  for i in range(0,Nelx-1):
     xG[i*quadpoints:(i+1)*quadpoints] = (2.*x[i]  + dx)/2. + zeta/2.*(dx)
  for i in range(0,Nely-1):
     yG[i*quadpoints:(i+1)*quadpoints] = (2.*y[i]  + dy)/2. + zeta/2.*(dy)
  for i in range(0,Nelz-1):
     zG[i*quadpoints:(i+1)*quadpoints] = (2.*z[i]  + dz)/2. + zeta/2.*(dz)

  return xG,yG,zG


def getGlobU(u):
  nvars,quadpoints,quadpoints,quadpoints,Nelx,Nely,Nelz = np.shape(u)
  uG = np.zeros((nvars,quadpoints*Nelx,quadpoints*Nely,quadpoints*Nelz))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
        for m in range(0,nvars):
          uG[m,i*quadpoints:(i+1)*quadpoints,j*quadpoints:(j+1)*quadpoints,k*quadpoints:(k+1)*quadpoints] = u[m,:,:,:,i,j,k]
  return uG


def f(x,y,z,qnum):
  return np.sin(z)



def TGVIC(x,y,z,qnum):
  order,order,order = np.shape(x)
  rho = 1.
  rho0 = 1.
  u = np.sin(x)*np.cos(y)*np.cos(z)
  v = -np.cos(x)*np.sin(y)*np.cos(z)
  w = 0
  speed_of_sound = 10.
  gamma = 1.4
  R = 287.
  p0 = speed_of_sound**2*gamma/rho0
  Cv = 5./2.*R
  p = p0 + rho0/16.*(np.cos(2.*x) + np.cos(2.*y) )*(np.cos(2.*z) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,order,order,order))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q[qnum]







def getIC(main,f,qnum):
  ## First perform integration in x
  for i in range(0,main.Npx):
    for j in range(0,main.Npy):
      for k in range(0,main.Npz):
        xloc = (main.x[i] + main.x[i] + main.dx)/2. + main.zeta/2.*(main.dx)
        yloc = (main.y[j] + main.y[j] + main.dy)/2. + main.zeta/2.*(main.dy)
        zloc = (main.z[k] + main.z[k] + main.dz)/2. + main.zeta/2.*(main.dz)
        for p in range(0,main.order):
          for q in range(0,main.order):
            for r in range(0,main.order):
              tg = np.ones((main.quadpoints,main.quadpoints,main.quadpoints))
              main.a.a[qnum,p,q,r,i,j,k] = (2.*p+1)*(2.*q+1)*(2.*r+1)/8.*np.sum(main.weights[:,None,None]*main.weights[None,:,None]*main.weights[None,None,:]*main.w[p][:,None,None]*main.w[q][None,:,None]*main.w[r][None,None,:]*\
                                           TGVIC(xloc[:,None,None]*tg,yloc[None,:,None]*tg,zloc[None,None,:]*tg,qnum))


L = 2.*np.pi
Nel = np.array([2**3,2**3,2**3])
order = 2
MZorder = 4
quadpoints = 2**2
mu = 0./1600.
x = np.linspace(0,L,Nel[0]+1)
y = np.linspace(0,L,Nel[1]+1)
z = np.linspace(0,L,Nel[2]+1)
t = 0
dt = 2.5e-3
et = 10.
iteration = 0
save_freq = 20
schemes = fschemes('central','IP')
eqns = equations('Navier-Stokes',schemes)
procx = 1
procy = 1
main = variables(Nel,order  ,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,schemes,'tau-model',procx,procy)
MZ   = variables(Nel,MZorder,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,schemes,'tau-model',procx,procy)

speed_of_sound = 10.
gamma = 1.4
R = 287.
p0 = speed_of_sound**2*gamma/1.

ainf = np.sqrt(p0/1.4)
MZ.tau = 1*L/Nel[0]/MZorder**2/ainf


xG,yG,zG = getGlobGrid(x,y,z,main.zeta)
np.savez('DGgrid',x=xG,y=yG,z=zG)
for qnum in range(0,eqns.nvars):
  getIC(main,f,qnum)
  if (main.mpi_rank == 0):
    print(qnum)

reconstructU(main,main.a)

if (main.mpi_rank == 0):
  if not os.path.exists('Solution'):
     os.makedirs('Solution')


#print(np.linalg.norm(main.u[0]))
t0 = time.time()
while (main.t <= main.et - main.dt/2):
  if (main.iteration%main.save_freq == 0):
    reconstructU(main,main.a)
    uG = gatherSolSlab(main,eqns,main.a)

    if (main.mpi_rank == 0):
      UG = getGlobU(uG)
      #uGF = getGlobU(uG)
      sys.stdout.write('======================================' + '\n')
      sys.stdout.write('wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('t = ' + str(main.t) +  '   rho sum = ' + str(np.sum(uG[0])) + '\n')
      np.savez('Solution/npsol' + str(main.iteration),U=UG,a=main.a.a,t=main.t,iteration=main.iteration,order=order)
      sys.stdout.flush()
      #plt.clf()
      #plt.plot(UG[1,0,:,0])
      #plt.contourf(uG[0,1,1,:,:],100)
      #xm = xG[:]#0.5*(x[0:-1] + x[1::])
      #plt.plot(xG[:],uG2[0,:,1])
      #plt.plot(xm,-1j*np.exp(1j*xm)*np.exp((-1j - mu)*main.t),'o',mfc='none')
      #plt.pause(0.0001)
  advanceSol(main,MZ,eqns,schemes)
reconstructU(main,main.a)
uG = gatherSolSlab(main,eqns,main.a)
if (main.mpi_rank == 0):
  print('Final Time = ' + str(time.time() - t0),'Sol Norm = ' + str(np.linalg.norm(uG)) )
