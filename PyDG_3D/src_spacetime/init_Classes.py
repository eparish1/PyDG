import numpy as np
import sys
from mpi4py import MPI
#from petsc4py import PETSc
from MPI_functions import getRankConnectionsSlab
from legendreBasis import *
from fluxSchemes import inviscidFlux,centralFluxGeneral
from navier_stokes import *
from linear_advection import *
#from equationFluxes import *
from DG_functions import getRHS
from turb_models import *
from boundary_conditions import *
from equations_class import *
from gas import *
from basis_class import *
from grid_functions import *
#from init_reacting_additions import add_reacting_to_main
class variable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz,Npt,arr_dtype='float64'):
      self.nvars = nvars
      self.order = order
      self.quadpoints = quadpoints
      self.a =np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.u =np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aR_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npy,Npz,Npt),dtype=arr_dtype)
      self.aL_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npy,Npz,Npt),dtype=arr_dtype)
      self.aU_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npz,Npt),dtype=arr_dtype)
      self.aD_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npz,Npt),dtype=arr_dtype)
      self.aF_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npt),dtype=arr_dtype)
      self.aB_edge = np.zeros((nvars,order[0],order[1],order[2],order[3],Npx,Npy,Npt),dtype=arr_dtype)


      self.uR_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt),dtype=arr_dtype)
      self.uL_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt),dtype=arr_dtype)
      self.uU_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt),dtype=arr_dtype)
      self.uD_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt),dtype=arr_dtype)
      self.uF_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt),dtype=arr_dtype)
      self.uB_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt),dtype=arr_dtype)

      self.uR = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uL = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uU = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uD = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uF = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uB = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)

      self.uFuture = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],Npx,Npy,Npz,Npt),dtype=arr_dtype)
#      self.uPast = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],Npx,Npy,Npz,Npt))


      self.aR = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aL = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aU = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aD = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aF = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.aB = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)

      self.uLS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uRS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uUS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uDS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uBS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
      self.uFS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)

class fluxvariable:
  def __init__(self,nvars,order,quadpoints,Npx,Npy,Npz,Npt,arr_dtype='float64'):
    self.nvars = nvars
    self.order = order
    self.quadpoints = quadpoints
    self.fx = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fy = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fz = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)

    self.fL = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fR = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fU = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fD = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fF = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fB = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)

    self.fRLS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx+1,Npy,Npz,Npt),dtype=arr_dtype)
    self.fUDS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy+1,Npz,Npt),dtype=arr_dtype)
    self.fFBS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz+1,Npt),dtype=arr_dtype)

    self.fRLI = np.zeros((nvars,order[1],order[2],order[3],Npx+1,Npy,Npz,Npt),dtype=arr_dtype)
    self.fUDI = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy+1,Npz,Npt),dtype=arr_dtype)
    self.fFBI = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz+1,Npt),dtype=arr_dtype)

    self.fRI = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fLI = np.zeros((nvars,order[1],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fUI = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fDI = np.zeros((nvars,order[0],order[2],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fFI = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fBI = np.zeros((nvars,order[0],order[1],order[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)



    self.fR_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt),dtype=arr_dtype)
    self.fL_edge = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npy,Npz,Npt),dtype=arr_dtype)
    self.fU_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt),dtype=arr_dtype)
    self.fD_edge = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npz,Npt),dtype=arr_dtype)
    self.fF_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt),dtype=arr_dtype)
    self.fB_edge = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npt),dtype=arr_dtype)

    ### Only need this for double flux schemes
    self.fRS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fUS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fFS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fLS = np.zeros((nvars,quadpoints[1],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fDS = np.zeros((nvars,quadpoints[0],quadpoints[2],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)
    self.fBS = np.zeros((nvars,quadpoints[0],quadpoints[1],quadpoints[3],Npx,Npy,Npz,Npt),dtype=arr_dtype)



class boundaryConditions:
  comm = MPI.COMM_WORLD
  mpi_rank = comm.Get_rank()
  def __init__(self,BC_type='periodic',BC_args=[]):
    check = 0
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    if (BC_type == 'periodic'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = periodic_bc
      self.args = BC_args
    if (BC_type == 'incompressible_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = incompwall_bc
      self.args = BC_args
    if (BC_type == 'isothermal_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = isothermalwall_bc
      self.args = BC_args
    if (BC_type == 'isothermal_wall_entropy'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = isothermalwall_entropy_bc
      self.args = BC_args
    if (BC_type == 'adiabatic_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = adiabaticwall_bc
      self.args = BC_args
    if (BC_type == 'dirichlet'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = dirichlet_bc
      self.args = BC_args
    if (BC_type == 'neumann'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = neumann_bc
      self.args = BC_args
    if (BC_type == 'reflecting_wall'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = reflectingwall_bc
      self.args = BC_args
    if (BC_type == 'reflecting_wall_x'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = reflectingwall_x_bc
      self.args = BC_args
    if (BC_type == 'slipwall_y'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = slipwall_y_bc
      self.args = BC_args
    if (BC_type == 'shuOscherBC'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = shuOscherBC
      self.args = BC_args
    if (BC_type == 'subsonic_outflow'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = subsonic_outflow
      self.args = BC_args
    if (BC_type == 'vishal_airfoil_viscous'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = vishal_airfoil_bc_viscous
      self.args = BC_args
    if (BC_type == 'vishal_airfoil_inviscid'):
      check = 1
      self.BC_type = BC_type
      self.applyBC = vishal_airfoil_bc_inviscid
      self.args = BC_args
    if (BC_type[0:6] == 'custom'):
      check = 1
      self.BC_type = BC_type 
      self.applyBC = globals()[BC_type]
      self.args = BC_args


    if (check == 0):
      if (mpi_rank == 0): print('BC type ' + BC_type + ' not found. PyDG quitting')
      sys.exit()

    
    

class variables:
  def __init__(self,Nel,order,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,procz,BCs,source,source_mag,shock_capturing,mol_str,basis_args,arr_dtype='float64'):
    ## DG scheme information
    self.basis_args = basis_args
    self.eq_str = eqns.eq_str
    self.Nel = Nel
    self.order = order
    self.quadpoints = quadpoints 
    self.t = t
    self.et = et
    self.dt = dt
    self.iteration = iteration
    self.save_freq = save_freq
    self.shock_capturing = shock_capturing
    ##============== MPI INFORMATION ===================
    self.procx = procx
    self.procy = procy
    self.procz = procz
    self.comm = MPI.COMM_WORLD
    self.num_processes = self.comm.Get_size()
    self.mpi_rank = self.comm.Get_rank()
    if (procx*procy*procz != self.num_processes):
      if (self.mpi_rank == 0):
        print('Error, correct x/y proc decomposition, now quitting!')
      sys.exit()
    self.Npy = int(float(Nel[1] / procy)) #number of points on each x plane. MUST BE UNIFORM BETWEEN PROCS
    self.Npx = int(float(Nel[0] / procx))

    ############################################
    self.Npz = int(float(Nel[2] / procz))
    ############################################
    
    self.Npt = Nel[-1]
    self.Nel_loc = np.array([self.Npx,self.Npy,self.Npz,self.Npt])
    
    ##############################################################
    self.sz = slice(int(self.mpi_rank)/int(self.procx*self.procy)*self.Npz,(int(self.mpi_rank)/int(self.procx*self.procy)+1)*self.Npz)
    self.sy = slice(int(int(self.mpi_rank)/int(self.procx)%self.procy)*self.Npy,(int(int(self.mpi_rank)/int(self.procx)%self.procy) + 1)*self.Npy)  ##slicing in y direction
    self.sx = slice(int(self.mpi_rank%self.procx)*self.Npx,int(self.mpi_rank%self.procx + 1)*self.Npx)
    self.rank_connect,self.BC_rank = getRankConnectionsSlab(self.mpi_rank,self.num_processes,self.procx,self.procy,self.procz)
    ##############################################################
    
    self.w,self.wp,self.wpedge,self.weights,self.zeta = gaussPoints(self.order[0],self.quadpoints[0])
    self.altarray = (-np.ones(self.order[0]))**(np.linspace(0,self.order[0]-1,self.order[0]))

    self.w0,self.wp0,self.wpedge0,self.weights0,self.zeta0 = gaussPoints(self.order[0],self.quadpoints[0])
    self.altarray0 = (-np.ones(self.order[0]))**(np.linspace(0,self.order[0]-1,self.order[0]))
    self.w1,self.wp1,self.wpedge1,self.weights1,self.zeta1 = gaussPoints(self.order[1],self.quadpoints[1])
    self.altarray1 = (-np.ones(self.order[1]))**(np.linspace(0,self.order[1]-1,self.order[1]))
    self.w2,self.wp2,self.wpedge2,self.weights2,self.zeta2 = gaussPoints(self.order[2],self.quadpoints[2])
    self.altarray2 = (-np.ones(self.order[2]))**(np.linspace(0,self.order[2]-1,self.order[2]))
    self.w3,self.wp3,self.wpedge3,self.weights3,self.zeta3 = gaussPoints(self.order[3],self.quadpoints[3])
    self.altarray3 = (-np.ones(self.order[3]))**(np.linspace(0,self.order[3]-1,self.order[3]))


    self.w0_c,self.wp0_c,self.wpedge0_c,self.weights0_c,self.zeta0_c = gaussPoints(self.order[0],self.order[0])
    self.altarray0_c = (-np.ones(self.order[0]))**(np.linspace(0,self.order[0]-1,self.order[0]))
    self.w1_c,self.wp1_c,self.wpedge1_c,self.weights1_c,self.zeta1_c = gaussPoints(self.order[1],self.order[1])
    self.altarray1_c = (-np.ones(self.order[1]))**(np.linspace(0,self.order[1]-1,self.order[1]))
    self.w2_c,self.wp2_c,self.wpedge2_c,self.weights2_c,self.zeta2_c = gaussPoints(self.order[2],self.order[2])
    self.altarray2_c = (-np.ones(self.order[2]))**(np.linspace(0,self.order[2]-1,self.order[2]))
    self.w3_c,self.wp3_c,self.wpedge3_c,self.weights3_c,self.zeta3_c = gaussPoints(self.order[3],self.order[3])
    self.altarray3_c = (-np.ones(self.order[3]))**(np.linspace(0,self.order[3]-1,self.order[3]))


    #xtmp,ytmp,ztmp = np.meshgrid(xG,yG,zG,indexing='ij')
    self.x,self.y,self.z = x,y,z

    Xtmp = np.zeros((3,Nel[0]+1,Nel[1]+1,Nel[2]+1))
    Xtmp[0],Xtmp[1],Xtmp[2] = x,y,z
    X_el = get_Xel(Xtmp,self.sx,self.sy,self.sz)
    self.J,self.Jinv,self.Jdet,self.J_edge_det,self.normals = computeJacobian(X_el,self.zeta0,self.zeta1,self.zeta2)
    self.xG,self.yG,self.zG = getGlobGrid(self,x,y,z,self.zeta0,self.zeta1,self.zeta2)
    self.Minv = getMassMatrix(self)
    self.gas = gasClass() 
    self.Cv = self.gas.Cv
    self.Cp = self.gas.Cp


    self.reacting = False
    ## Initialize BCs
    self.BCs = BCs
    self.rightBC = boundaryConditions(BCs[0],BCs[1])
    self.topBC = boundaryConditions(BCs[2],BCs[3])
    self.leftBC = boundaryConditions(BCs[4],BCs[5])
    self.bottomBC = boundaryConditions(BCs[6],BCs[7])
    self.frontBC = boundaryConditions(BCs[8],BCs[9])
    self.backBC = boundaryConditions(BCs[10],BCs[11])

    self.cgas = False 
    self.cgas_field = False 
    self.cgas_field_LR = False
    self.cgas_field_L = False
    self.cgas_field_R = False
    self.cgas_field_UD = False
    self.cgas_field_U =  False
    self.cgas_field_D = False 
    self.cgas_field_FB = False 
    self.cgas_field_F = False
    self.cgas_field_B = False 


    self.cgas_field_L_edge = False 
    self.cgas_field_R_edge = False 
    self.cgas_field_D_edge = False 
    self.cgas_field_U_edge = False 
    self.cgas_field_B_edge = False 
    self.cgas_field_F_edge = False 





    ## Sources
    self.fsource = source
    self.source_mag = source_mag
    ## Initialize arrays
    self.dx = x[1,0,0] - x[0,0,0]
    self.dy = y[0,1,0] - y[0,0,0]
    self.dz = z[0,0,1] - z[0,0,0]
    self.dx2 = np.diff(x[:,0,0])[self.sx]
    self.dy2 = np.diff(y[0,:,0])[self.sy]
    self.dz2 = np.diff(z[0,0,:])

    #print(np.shape(xG))
    #self.x = x[self.sx]
    #self.y = y[self.sy] 
    #self.z = z
    self.nvars = eqns.nvars

    self.a0 = np.zeros((eqns.nvars,self.order[0],self.order[1],self.order[2],self.order[3],self.Npx,self.Npy,self.Npz,self.Npt),dtype=arr_dtype)
    self.a = variable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt,arr_dtype)

    #self.adum = variable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt)

    self.b = variable(eqns.nvisc_vars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt,arr_dtype)

    self.iFlux = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt,arr_dtype)
    self.vFlux = fluxvariable(eqns.nvisc_vars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt,arr_dtype)
    self.vFlux2 = fluxvariable(eqns.nvars,self.order,self.quadpoints,self.Npx,self.Npy,self.Npz,self.Npt,arr_dtype)

    self.mus = mu
    self.mu = mu#np.ones(np.append( eqns.nmus, np.shape( self.a.u[0])))*self.mus
#    self.muR = np.ones(np.append( eqns.nmus, np.shape( self.a.uR[0])))*self.mus
#    self.muL = np.ones(np.append( eqns.nmus, np.shape( self.a.uL[0])))*self.mus
#    self.muU = np.ones(np.append( eqns.nmus, np.shape( self.a.uU[0])))*self.mus
#    self.muD = np.ones(np.append( eqns.nmus, np.shape( self.a.uD[0])))*self.mus
#    self.muF = np.ones(np.append( eqns.nmus, np.shape( self.a.uF[0])))*self.mus
#    self.muB = np.ones(np.append( eqns.nmus, np.shape( self.a.uB[0])))*self.mus
#    self.mu0 = np.ones(np.append( eqns.nmus, np.shape( self.a.u[0] )))*self.mus
#    self.mu0R =np.ones(np.append( eqns.nmus, np.shape( self.a.uR[0])))*self.mus
#    self.mu0L =np.ones(np.append( eqns.nmus, np.shape( self.a.uL[0])))*self.mus
#    self.mu0U =np.ones(np.append( eqns.nmus, np.shape( self.a.uU[0])))*self.mus
#    self.mu0D =np.ones(np.append( eqns.nmus, np.shape( self.a.uD[0])))*self.mus
#    self.mu0F =np.ones(np.append( eqns.nmus, np.shape( self.a.uF[0])))*self.mus
#    self.mu0B =np.ones(np.append( eqns.nmus, np.shape( self.a.uB[0])))*self.mus

#    self.tmp0 = np.zeros(np.shape(np.rollaxis(np.tensordot(self.w0*self.weights0[None,:],self.iFlux.fx,axes=([1],[1])),0,9)))
#    self.tmp1 = np.zeros(np.shape(np.rollaxis(np.tensordot(self.w1*self.weights1[None,:],self.tmp0,axes=([1],[1])),0,9)))
#    self.tmp2 = np.zeros(np.shape(np.rollaxis(np.tensordot(self.w2*self.weights2[None,:],self.tmp1,axes=([1],[1])),0,9)))
#    self.tmp3 = np.zeros(np.shape(np.rollaxis(np.tensordot(self.w3*self.weights3[None,:],self.tmp2,axes=([1],[1])),0,9)))

    self.RHS = np.zeros((eqns.nvars,self.order[0],self.order[1],self.order[2],self.order[3],self.Npx,self.Npy,self.Npz,self.Npt),dtype=arr_dtype)
    self.NLiter = 0 
    self.linear_iteration = 0
    ### Check turbulence models
    self.turb_str = turb_str
    check = 0
    if (turb_str == 'Orthogonal Dynamics'):
      self.getRHS = orthogonalDynamics
      check = 1
    if (turb_str == 'tau-model'):
      self.getRHS = tauModelLinearized
      check = 1
    if (turb_str == 'orthogonal subscale'):
      self.getRHS = orthogonalSubscale
      check = 1
    if (turb_str == 'orthogonal subscale POD'):
      self.getRHS = orthogonalSubscale_POD
      self.PLQLu = np.zeros(np.shape(self.a.a))
      check = 1
    if (turb_str == 'LSPG POD'):
      self.getRHS = LSPG_POD
      self.PLQLu = np.zeros(np.shape(self.a.a))
      check = 1
    if (turb_str == 'dynamic tau entropy'):
      self.getRHS = dynamicTauEntropy
      check = 1
    if (turb_str == 'orthogonal subscale entropy'):
      self.getRHS = orthogonalSubscaleEntropy
      check = 1
    if (turb_str == 'tau-modelFD'):
      self.getRHS = tauModelFD
      if (self.mpi_rank == 0): print('Using finite difference tau model')
      check = 1
    if (turb_str == 'tau-modelFDEntropy'):
      self.getRHS = tauModelFDEntropy
      check = 1
    if (turb_str == 'FM1'):
      self.getRHS = FM1Linearized 
      check = 1
    if (turb_str == 'Smagorinsky'):
      self.getRHS = DNS
      if (self.mpi_rank == 0): print('Using Smagorinsky Model')
    if (check == 0):
      self.getRHS = DNS
#      if (self.mpi_rank == 0):
#         print('Error, turb model ' + turb_str + 'not found. Setting to DNS')
    else:
      if (self.mpi_rank == 0):
         print('Using turb model ' + turb_str)

    self.basis = basis_class('Legendre',basis_args)
    self.mol_str = mol_str
    if (eqns.eq_str[0:-2] == 'Navier-Stokes Reacting'):
      self = add_reacting_to_main(self,mol_str)
    if (eqns.eq_str == 'Navier-Stokes Entropy'):
      norder = order[0]*order[1]*order[2]*order[3]
      self.EMM = np.zeros((norder*5,norder*5,self.Npx,self.Npy,self.Npz,self.Npt) )
