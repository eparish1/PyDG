import numpy as np
from init_Classes import *
from DG_functions import reconstructU
from MPI_functions import gatherSol
import matplotlib.pyplot as plt
from timeSchemes import advanceSol
import time


def getGlobGrid(x,y,zeta):
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  Nelx,Nely = np.size(x),np.size(y)
  order = np.size(zeta)
  xG = np.zeros((np.size(x)*np.size(zeta)))
  yG = np.zeros((np.size(y)*np.size(zeta)))
  for i in range(0,Nelx):
     xG[i*order:(i+1)*order] = (2.*x[i]  + dx)/2. + zeta/2.*(dx)
  for i in range(0,Nely):
     yG[i*order:(i+1)*order] = (2.*y[i]  + dy)/2. + zeta/2.*(dy)
  return xG,yG

def getGlobU(u):
  nvars,order,order,Nelx,Nely = np.shape(u)
  uG = np.zeros((nvars,order*Nelx,order*Nely))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for m in range(0,nvars):
        uG[m,i*order:(i+1)*order,j*order:(j+1)*order] = u[m,:,:,i,j]
  return uG

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
  nx,ny = np.shape(x)
  q = np.zeros((1,nx,ny))
  q[0] = np.cos(x)*np.cos(y) - np.sin(x)*np.sin(y)
  return q[qnum]



L = 2.*np.pi
L = 2.*np.pi

Na = np.array([8,16,32,64,128])
Oa = np.array([1,2,3,4,5])
error = np.zeros((5,5))
dtA = np.array([0.01,5.e-3,2.e-3,1.e-3,5.e-4])
for ii in range(0,5):
 for jj in range(0,5):
  Nel = np.array([Na[ii],Na[ii]])
  order = Oa[jj]
  nu = 1.e-1
  x = np.linspace(0,L,Nel[0]+1)
  y = np.linspace(0,L,Nel[1]+1)
  t = 0
  dt = 5.e-4#dtA[ii]/float(jj+1.)
  et = 2.
  iteration = 0
  save_freq = 100
  eqns = equations('Diffusion')
  main = variables(Nel,order,eqns,nu,x,y,t,et,dt,iteration,save_freq)
  schemes = fschemes('central','central')
  xG,yG = getGlobGrid(x[0:-1],y[0:-1],main.zeta) 
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
        #plt.contourf(uG[0,0,0,:,:],100)
        plt.plot(0.5*(x[0:-1] + x[1::]),uG[0,0,0,:,4])
        xm = 0.5*(x[0:-1] + x[1::])
        ym = 0.5*(y[0:-1] + y[1::])
        plt.plot(xm,np.exp(-2.*nu*main.t)*(np.cos(xm[:,None])*np.cos(ym[None,:]) - np.sin(xm[:,None])*np.sin(ym[None,:]))[:,4],'o') 
        plt.pause(0.0001)
    advanceSol(main,eqns,schemes)
  print('Final Time = ' + str(time.time() - t0))
  reconstructU(main,main.a)
  uG = gatherSol(main,eqns,main.a)
  uB = getGlobU(uG)
  error[ii,jj] = np.mean( abs( np.exp(-2.*nu*main.t)*(np.cos(xG[:,None])*np.cos(yG[None,:]) - np.sin(xG[:,None])*np.sin(yG[None,:]))  - uB[0])) 
  print('error = ' + str(error))

