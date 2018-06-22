import matplotlib.pyplot as plt
import numpy as np
from MPI_functions import gatherSolSpectral,globalDot
from equations_class import *
from tensor_products import *
from navier_stokes import strongFormEulerXYZ
from jacobian_schemes import computeJacobian_full_pod
def orthogonalDynamics(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.ones(np.shape(main.a.a))
    filtarray[:,main.rorder::] = 0.
    a0 = np.zeros(np.shape(main.a.a))
    a0[:] = main.a.a[:]
    eqns.getRHS(main,MZ,eqns)
    RHS1 = np.zeros(np.shape(main.RHS))
    RHS1[:] = main.RHS[:]
    main.a.a[:] = a0[:]*filtarray[:]
    eqns.getRHS(main,MZ,eqns)
    RHS2 = np.zeros(np.shape(main.RHS))
    RHS2[:] = main.RHS[:]
    main.RHS[:] = RHS1[:] - RHS2[:]
    main.a.a[:] = a0[:]


def DNS(regionManager,eqns):
  regionManager.getRHS_REGION_INNER(regionManager,eqns)

def projection(main,U):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  return a_project

def orthogonalProjection(main,U):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  U_project = main.basis.reconstructUGeneral(main,a_project)
  U_orthogonal = U - U_project
  return U_orthogonal



def orthogonalSubscale(regionManager,eqns):
   #eqns.getRHS(main,MZ,eqns)
   regionManager.getRHS_REGION_INNER(regionManager,eqns)

   for region in regionManager.region:
     region.R0 = np.zeros(np.shape(region.RHS))
     region.R1 = np.zeros(np.shape(region.RHS))
     region.R0[:] = region.RHS[:]

     region.RHS[:] = 0.
     region.R = eqns.strongFormResidual(region,region.a.a,None)
     region.R_orthogonal= orthogonalProjection(region,region.R)

     PLQLu2 = np.zeros(np.shape(region.RHS))
     region.u0 = region.a.u*1.

     region.a.u[:] = region.u0[:]
     eqns.evalFluxXYZLin(region,region.a.u,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,[-region.R_orthogonal])
     region.basis.applyVolIntegral(region,region.iFlux.fx,region.iFlux.fy,region.iFlux.fz,PLQLu2)
     region.RHS[:] = region.R0[:] 
     region.tau = 100.
     indx = region.tau*abs(PLQLu2[4]) > (  abs(region.R0[4]) + 1e-3)
     region.a.a[:,indx] = 0.
     region.RHS[:,indx] = 0.
#   for i in range(main.order[0]-1,0,-1):
#     indx = np.ones(np.shape(PLQLu2[4,0]),dtype=bool)  #initialize an array with all trues
#     for j in range(main.order[0] - 1, i-1,-1):
#       chk = 100.*abs(PLQLu2[4,j]) > (  abs(R0[4,j]) + 1e-3)
#       indx = indx & chk 
#     main.a.a[0,i,indx] = 0.
#     main.RHS[0,i,indx] = 0.
#     main.a.a[1,i,indx] = 0.
#     main.RHS[1,i,indx] = 0.
#     main.a.a[2,i,indx] = 0.
#     main.RHS[2,i,indx] = 0.
#     main.a.a[3,i,indx] = 0.
#     main.RHS[3,i,indx] = 0.
#     main.a.a[4,i,indx] = 0.
#     main.RHS[4,i,indx] = 0.





def orthogonalSubscale2(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]


   eps = 1.e-5
   main.RHS[:] = 0.
   fx,fy,fz = strongFormEulerXYZ(main,main.a.a,None)
   #u0 = main.a.u*1.
   f = fx + fy + fz
   #main.a.u[:] = u0[:]
   f_orthogonal = orthogonalProjection(main,f)
   evalFluxXYZEulerLin(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-f_orthogonal])
   PLQLu2 = np.zeros(np.shape(main.RHS))
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
   #print(np.linalg.norm(PLQLu2 - PLQLu),np.linalg.norm(PLQLu[1]),np.linalg.norm(PLQLu2[1]))
   #print(np.linalg.norm(PLQLu),np.linalg.norm(f),np.linalg.norm(f_orthogonal),np.linalg.norm(R1),np.linalg.norm(R2 - R1))
   tau = 1. #tau8.0
   main.RHS[:] = R0[:] + tau*PLQLu2

## Evaluate the tau model through the FD approximation. This is expensive AF
def tauModelFDEntropy(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    #eqns.getRHS(main,main,eqns)
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    Rtmp = np.zeros(np.shape(MZ.RHS))
    RHS1[:] =MZ.RHS[:]
    #Rtmp[:] = RHS1[:]
    #MZ.basis.applyMassMatrix(MZ,Rtmp)
    #MZ.a.a[:] = 0.
    #MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    #MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]
    #eqns.getRHS(MZ,MZ,eqns)
    #RHS2 = np.zeros(np.shape(MZ.RHS))
    #RHS2[:] = MZ.RHS[:]
    #MZ.a.a[:] = 0.
    #MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    #MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]*filtarray
    #eqns.getRHS(MZ,MZ,eqns)
    #RHS3 = np.zeros(np.shape(MZ.RHS))
    #RHS3[:] = MZ.RHS[:]
    #PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/(eps + 1e-30)
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ main.dx/MZ.order[0]**2*PLQLU


## Evaluate the tau model for only volume residual through the FD approximation. This is expensive AF
def tauModelFD(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    #eqns.getRHS(main,main,eqns)
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    Rtmp = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    Rtmp[:] = RHS1[:]
    MZ.basis.applyMassMatrix(MZ,Rtmp)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/(eps + 1e-30)
    tau = main.dx/MZ.order[0]**2
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + 0.1*PLQLU


## Evaluate the tau model through the FD approximation. This is expensive AF
def tauModelFD(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    #eqns.getRHS(main,main,eqns)
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    Rtmp = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    Rtmp[:] = RHS1[:]
    MZ.basis.applyMassMatrix(MZ,Rtmp)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/(eps + 1e-30)
    tau = main.dx/MZ.order[0]**2
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + 0.1*PLQLU


def FM1Linearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   tau = 0.1
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[0:5] =  RHS1[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] + main.a.a[5::] 
   main.RHS[5::] = -2.*main.a.a[5::]/tau + 2.*PLQLU
   main.comm.Barrier()




def tauModelLinearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   RHS1f_phys = main.basis.reconstructUGeneral(MZ,RHS1f)
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid'),'DNS' )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqnsLin.getRHS(MZ,MZ,eqnsLin,[RHS1f],[RHS1f_phys]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + main.dx/MZ.order[0]**2*PLQLU
   main.comm.Barrier()




def validateLinearized(main,MZ,eqns):
    # validation of the linearized equations
    # should have R(a + eps f) - R(a) / eps = Rlin(a)
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    eqnsLin.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHSLin = np.zeros(np.shape(MZ.RHS))
    RHSLin[:] = MZ.RHS[:]
    # now we need to compute the linearized RHS via finite difference
    eps = 1.e-6
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,eqns,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*MZ.a.a[:]
    eqns.getRHS(MZ,eqns,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    RHSLinFD = (RHS2 - RHS1)/eps
    print(np.linalg.norm(RHSLin),np.linalg.norm(RHSLinFD) )
    print(np.linalg.norm(RHSLin[:] -  RHSLinFD[:]) )
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]


def tauModelValidateLinearized(main,MZ,eqns):
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1f = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
    Z = reconstructUGeneral(main,main.a.a)
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    # now we need to compute the linearized RHS
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
    PLQLULin = np.zeros(np.shape(MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]))
    PLQLULin[:] = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

#    print(np.linalg.norm(PLQLU[0]))

    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLUFD = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/eps

    print(np.linalg.norm(PLQLULin),np.linalg.norm(PLQLUFD) )
    print(np.linalg.norm(PLQLULin[4] - PLQLUFD[4] ))
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

    plt.clf()
    plt.plot(PLQLULin[4,1,0,0,:,0,0])
    plt.plot( PLQLUFD[4,1,0,0,:,0,0],'o')
    plt.pause(0.001)
    main.comm.Barrier()



def DtauModel(main,MZ,eqns,schemes):

  def sendScalar(scalar,main):
    if (main.mpi_rank == 0):
      for i in range(1,main.num_processes):
        loc_rank = i
        main.comm.Send(np.ones(1)*scalar,dest=loc_rank,tag=loc_rank)
      return scalar
    else:
      test = np.ones(1)*scalar
      main.comm.Recv(test,source=0,tag=main.mpi_rank)
      return test[0]

  ### EVAL RESIDUAL AND DO MZ STUFF
  filtarray = np.zeros(np.shape(MZ.a.a))
  filtarray[:,0:main.order,0:main.order,:,:] = 1.
  eps = 1.e-5
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS1 = np.zeros(np.shape(MZ.RHS))
  RHS1[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS2 = np.zeros(np.shape(MZ.RHS))
  RHS2[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
  MZ.getRHS(MZ,eqns,schemes)
  RHS3 = np.zeros(np.shape(MZ.RHS))
  RHS3[:] = MZ.RHS[:]
  PLQLU = (RHS2[:,0:main.order,0:main.order] - RHS3[:,0:main.order,0:main.order])/eps

  if (main.rkstage == 0):
    ### Now do dynamic procedure to get tau
    filtarray2 = np.zeros(np.shape(MZ.a.a))
    filtarray2[:,0:MZ.forder,0:MZ.forder,:,:] = 1.
    eps = 1.e-5
    ## Get RHS
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.getRHS(MZ,eqns,schemes)
    RHS4 = np.zeros(np.shape(MZ.RHS))
    RHS4[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHS)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS5 = np.zeros(np.shape(MZ.RHS))
    RHS5[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHSf)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]*filtarray2
    MZ.getRHS(MZ,eqns,schemes)
    RHS6 = np.zeros(np.shape(MZ.RHS))
    RHS6[:] = MZ.RHS[:]
  
    ## Now compute PLQLUf
    PLQLUf = (RHS5[:,0:main.order,0:main.order] - RHS6[:,0:main.order,0:main.order])/eps
  
    PLQLUG = gatherSolSpectral(PLQLU,main)
    MZ.PLQLUG = PLQLUG
    PLQLUfG = gatherSolSpectral(PLQLUf[:,0:main.order,0:main.order],main)
    RHS1G = gatherSolSpectral(RHS1[:,0:main.order,0:main.order],main)
    RHS4G = gatherSolSpectral(RHS4[:,0:main.order,0:main.order],main)

    afG = gatherSolSpectral(main.a.a[:,0:main.order,0:main.order],main)

    if (main.mpi_rank == 0):
      num = 2.*np.mean(np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(RHS4G[1:3,0:MZ.forder,0:MZ.forder] - RHS1G[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      den =  np.mean ( np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(PLQLUG[1:3,0:MZ.forder,0:MZ.forder] - \
                                         (main.order/MZ.forder)*PLQLUfG[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      tau = num/(den + 1.e-1)
      print(tau)
    else:
      tau = 0.
    MZ.tau = np.maximum(0.,sendScalar(tau,main))
    #MZ.tau = sendScalar(tau,main)
  return 0.0002*PLQLU



def tauModel2(main,MZ,eqns,schemes):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.ones(np.shape(main.a.a))
    filtarray[:,main.rorder::,main.rorder::,:,:] = 0.
    eps = 1.e-5
    MZ.a.a[:] = main.a.a[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:]*filtarray + eps*RHS1[:]
    main.getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:] + eps*RHS1[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2 - RHS3)/eps
    return MZ.tau*PLQLU



def shockCapturingSetViscosity(main):
  ### Shock capturing
  filta = np.zeros(np.shape(main.a.a))
  filta[:,0:main.order[0]-1,main.order[1]-1,main.order[2]-1] = 1.
  af = main.a.a*filta    #make filtered state
  uf = reconstructUGeneral(main,af)
  udff = (main.a.u - uf)**2
  # now compute switch
  Se_num = volIntegrate(main.weights0,main.weights1,main.weights2,udff) 
  Se_den = volIntegrate(main.weights0,main.weights1,main.weights2,main.a.u**2)
  Se = (Se_num + 1e-10)/(Se_den + 1.e-30)
  eps0 = 1.*main.dx/main.order[0]
  s0 =1./main.order[0]**4
  kap = 5.
  se = np.log10(Se)
  #print(np.amax(udff))
  epse = eps0/2.*(1. + np.sin(np.pi/(2.*kap)*(se - s0) ) )
  epse[se<s0-kap] = 0.
  epse[se>s0  + kap] = eps0
  #plt.clf()
  #print(np.amax(epse),np.amin(epse))
  #plt.plot(epse[0,:,0,0])
  #plt.ylim([1e-9,0.005])
  #plt.pause(0.001)
  #print(np.shape(main.mu),np.shape(epse) )
  main.mu = main.mu0 + epse[0]
  main.muR = main.mu0R + epse[0]
  main.muL = main.mu0L + epse[0]
  main.muU = main.mu0F + epse[0]
  main.muD = main.mu0D + epse[0]
  main.muF = main.mu0U + epse[0]
  main.muB = main.mu0B + epse[0]




#########################################
## POD FUNCTIONS
#==============================
def projection_pod(u,V,regionManager):
  tmp = globalDot(V.transpose(),u,regionManager)
  u_proj = np.dot(V, tmp)
  return u_proj

## Function for orthogonal subscale MZ tau model
def orthogonalSubscale_POD(regionManager,eqns):

  if (regionManager.iteration%10 == 0 and regionManager.rk_stage == 0):
    J = computeJacobian_full_pod(regionManager,eqns)
    e,s = np.linalg.eig(J)
    tau = 0.2/np.amax(abs(e))
    regionManager.tau = tau 
    if (regionManager.mpi_rank == 0):
      print('Spectral Radius = ',np.amax(abs(e)))
      print('tau             = ',np.amax(abs(tau)))

  eps = 1e-5
  a0 = regionManager.a*1.
  regionManager.getRHS_REGION_INNER(regionManager,eqns) #includes loop over all regions
  RHS0 = regionManager.RHS*1.

  #==================================================
  R_ortho = RHS0 -  projection_pod(RHS0,regionManager.V,regionManager)

  regionManager.a[:] = a0[:] + eps*R_ortho

  regionManager.getRHS_REGION_INNER(regionManager,eqns) #includes loop over all regions
  #main.basis.applyMassMatrix(main,main.RHS)
  PLQLu = (regionManager.RHS - RHS0)/eps
  #regionManager.PLQLu[:] = PLQLu
  #tau = regionManager.region[0].tau
  tau = regionManager.tau
  #=====================================
  regionManager.RHS[:] =  RHS0[:]+ tau*PLQLu
  regionManager.a[:] = a0[:]
