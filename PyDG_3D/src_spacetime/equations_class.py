import numpy as np
from mpi4py import MPI
import sys
from navier_stokes import *
from navier_stokes_reacting import *

from linear_advection import *
from DG_functions import getRHS
class equations:
  def __init__(self,eq_str,schemes,turb_str):
    comm = MPI.COMM_WORLD
    self.turb_str = turb_str
    mpi_rank = comm.Get_rank()
    iflux_str = schemes[0]
    vflux_str = schemes[1]
    check_eq = 0
    if (eq_str == 'FM1 Navier-Stokes'):
      self.nmus = 1
      self.eq_str = eq_str
      check_eq = 1
      self.nvars = 10
      self.evalFluxX = evalFluxXEuler 
      self.evalFluxY = evalFluxYEuler
      self.evalFluxZ = evalFluxZEuler
      ## select appopriate flux scheme
      checki = 0
      if (iflux_str == 'central'):
        self.inviscidFlux = eulerCentralFlux
        checki = 1
      if (iflux_str == 'roe'):
        self.inviscidFlux = kfid_roeflux
        checki = 1
      if (iflux_str == 'rusanov'):
        self.inviscidFlux = rusanovFlux
        checki = 1
      if (checki == 0):
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not valid. Options are "central", "roe", "rusanov". PyDG quitting')
        sys.exit()
      checkv = 0 
      if (vflux_str == 'BR1'):
        print('Error, BR1 not completed for 3D. PyDG quitting')
        sys.exit()
        self.evalViscousFluxX = evalViscousFluxXNS_BR1
        self.evalViscousFluxY = evalViscousFluxYNS_BR1
        self.evalTauFluxX = evalTauFluxXNS_BR1
        self.evalTauFluxY = evalTauFluxYNS_BR1
        self.vflux_type = 'BR1'
        checkv = 1
        self.viscousFlux = centralFlux
      if (vflux_str == 'IP'):
        self.getRHS = getRHS
        self.evalViscousFluxX = evalViscousFluxXNS_IP
        self.evalViscousFluxY = evalViscousFluxYNS_IP
        self.evalViscousFluxZ = evalViscousFluxZNS_IP
        self.getGs = getGsNS 
        self.getGsX = getGsNSX_FAST
        self.getGsY = getGsNSY_FAST
        self.getGsZ = getGsNSZ_FAST
        self.vflux_type = 'IP'
        checkv = 1
      if (vflux_str == 'Inviscid'):
        self.getRHS = getRHS_INVISCID
        self.vflux_type = 'Inviscid'
        checkv = 1
      if (checkv == 0):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not valid. Options are, "IP", "Inviscid". PyDG quitting')
        sys.exit() 
     

    if (eq_str[0:-2] == 'Navier-Stokes Reacting'):
      nscalars = int(eq_str[-1])
      self.nmus = 1 + nscalars
      self.eq_str = eq_str
      check_eq = 1
      self.nvars = 5 + nscalars
      self.evalFluxX = evalFluxXEuler_reacting 
      self.evalFluxY = evalFluxYEuler_reacting
      self.evalFluxZ = evalFluxZEuler_reacting
      ## select appopriate flux scheme
      checki = 0
      if (iflux_str == 'central'):
        self.inviscidFlux = eulerCentralFlux_reacting
        checki = 1
      if (iflux_str == 'roe'):
        self.inviscidFlux = kfid_roeflux_reacting
        checki = 1
      if (iflux_str == 'rusanov'):
        self.inviscidFlux = rusanovFlux_reacting
        checki = 1
      if (checki == 0):
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not valid. Options are "central", "roe", "rusanov". PyDG quitting')
        sys.exit()
      checkv = 0 
      if (vflux_str == 'BR1'):
        print('Error, BR1 not completed for 3D. PyDG quitting')
        sys.exit()
        checkv = 1
        self.viscousFlux = centralFlux
      if (vflux_str == 'IP'):
        self.viscous = True
        self.getRHS = getRHS
        self.evalViscousFluxX = evalViscousFluxXNS_IP_reacting
        self.evalViscousFluxY = evalViscousFluxYNS_IP_reacting
        self.evalViscousFluxZ = evalViscousFluxZNS_IP_reacting
        self.getGsX = getGsNSX_FAST_reacting
        self.getGsY = getGsNSY_FAST_reacting
        self.getGsZ = getGsNSZ_FAST_reacting
        self.vflux_type = 'IP'
        checkv = 1
      if (vflux_str == 'Inviscid'):
        self.viscous = False
        self.getRHS = getRHS
        self.vflux_type = 'Inviscid'
        checkv = 1
      if (checkv == 0):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not valid. Options are, "IP", "Inviscid". PyDG quitting')
        sys.exit() 



    if (eq_str == 'Navier-Stokes'):
      self.nmus = 1
      self.eq_str = eq_str
      check_eq = 1
      self.nvars = 5
      self.nvisc_vars = 5
      self.evalFluxX = evalFluxXEuler 
      self.evalFluxY = evalFluxYEuler
      self.evalFluxZ = evalFluxZEuler
      ## select appopriate flux scheme
      checki = 0
      if (iflux_str == 'central'):
        self.inviscidFlux = eulerCentralFlux
        checki = 1
      if (iflux_str == 'roe'):
        self.inviscidFlux = kfid_roeflux
        checki = 1
      if (iflux_str == 'rusanov'):
        self.inviscidFlux = rusanovFlux
        checki = 1
      if (checki == 0):
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not valid. Options are "central", "roe", "rusanov". PyDG quitting')
        sys.exit()
      checkv = 0 
      if (vflux_str == 'BR1'):
        print('Error, BR1 not completed for 3D. PyDG quitting')
        sys.exit()
        self.evalViscousFluxX = evalViscousFluxXNS_BR1
        self.evalViscousFluxY = evalViscousFluxYNS_BR1
        self.evalTauFluxX = evalTauFluxXNS_BR1
        self.evalTauFluxY = evalTauFluxYNS_BR1
        self.vflux_type = 'BR1'
        checkv = 1
        self.viscousFlux = centralFlux
      if (vflux_str == 'IP'):
        self.viscous = True
        self.getRHS = getRHS
        self.evalViscousFluxX = evalViscousFluxXNS_IP
        self.evalViscousFluxY = evalViscousFluxYNS_IP
        self.evalViscousFluxZ = evalViscousFluxZNS_IP
        self.getGs = getGsNS 
        self.getGsX = getGsNSX_FAST
        self.getGsY = getGsNSY_FAST
        self.getGsZ = getGsNSZ_FAST
        self.vflux_type = 'IP'
        checkv = 1
      if (vflux_str == 'Inviscid'):
        self.viscous = False
        self.getRHS = getRHS
        self.vflux_type = 'Inviscid'
        checkv = 1
      if (checkv == 0):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not valid. Options are, "IP", "Inviscid". PyDG quitting')
        sys.exit() 

    if (eq_str == 'Linearized Navier-Stokes'):
      self.getRHS = getRHS
      check_eq = 1
      self.nvars = 5
      self.nvisc_vars = 5
      self.evalFluxX = evalFluxXEulerLin 
      self.evalFluxY = evalFluxYEulerLin
      self.evalFluxZ = evalFluxZEulerLin
      self.viscous = False
      ## select appopriate flux scheme
      checki = 0
      if (iflux_str == 'central'):
        self.inviscidFlux = eulerCentralFluxLinearized
        checki = 1
      if (iflux_str == 'roe'):
        self.inviscidFlux = kfid_roeflux
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not yet implemented for ' + eq_str + '. Options are "central". PyDG quitting')
        sys.exit()
        checki = 1
      if (iflux_str == 'rusanov'):
        self.inviscidFlux = rusanovFlux
        self.inviscidFlux = kfid_roeflux
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not yet implemented for ' + eq_str + '. Options are "central". PyDG quitting')
        sys.exit()
        checki = 1
      if (checki == 0):
        if (mpi_rank == 0): print('Error, inviscid flux scheme ' + iflux_str + ' not yet implemented for ' + eq_str + '. Options are "central". PyDG quitting')
        sys.exit()
      checkv = 0 
      if (vflux_str == 'BR1'):
        print('Error, BR1 not completed for 3D. PyDG quitting')
        sys.exit()
        self.evalViscousFluxX = evalViscousFluxXNS_BR1
        self.evalViscousFluxY = evalViscousFluxYNS_BR1
        self.evalTauFluxX = evalTauFluxXNS_BR1
        self.evalTauFluxY = evalTauFluxYNS_BR1
        self.vflux_type = 'BR1'
        checkv = 1
        self.viscousFlux = centralFlux
      if (vflux_str == 'IP'):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not yet implemented for ' + eq_str + '. Options are "INVISCID". PyDG quitting')
        sys.exit()
        self.viscous = True
        self.evalViscousFluxX = evalViscousFluxXNS_IP
        self.evalViscousFluxY = evalViscousFluxYNS_IP
        self.evalViscousFluxZ = evalViscousFluxZNS_IP
        self.getGs = getGsNS 
        self.getGsX = getGsNSX_FAST
        self.getGsY = getGsNSY_FAST
        self.getGsZ = getGsNSZ_FAST
        self.vflux_type = 'IP'
        checkv = 1
      if (vflux_str == 'Inviscid'):
        self.vflux_type = 'Inviscid'
        checkv = 1
      if (checkv == 0):
        if (mpi_rank == 0): print('Error, viscous flux scheme ' + vflux_str + ' not valid. Options are, "IP", "Inviscid". PyDG quitting')
        sys.exit() 



    if (eq_str == 'Linear-Advection'):
      check_eq = 1
      #print('Linear-Advection is not yet complete for 3D, PyDG quitting')
      #sys.exit()
      self.nvars = 1
      self.nvisc_vars = 2
      self.viscous = True
      self.getRHS = getRHS 
      self.evalFluxX = evalFluxXLA
      self.evalFluxY = evalFluxYLA
      self.evalFluxZ = evalFluxZLA
      self.inviscidFlux = linearAdvectionCentralFlux

      if (vflux_str == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.evalViscousFluxZ = evalViscousFluxZLA_IP
        self.getGsX = getGsD_X
        self.getGsY = getGsD_Y
        self.getGsZ = getGsD_Z


      if (vflux_str == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1

    if (eq_str == 'Diffusion'):
      check_eq = 1
      self.viscous = True
      self.getRHS = getRHS 
      self.inviscidFlux = diffusionCentralFlux
      self.nvars = 1
      self.nvisc_vars = 2
      self.evalFluxX = evalFluxD
      self.evalFluxY = evalFluxD
      self.evalFluxZ = evalFluxD
      #self.getEigs = getEigsEuler
      if (vflux_str == 'IP'):
        self.evalViscousFluxX = evalViscousFluxXLA_IP
        self.evalViscousFluxY = evalViscousFluxYLA_IP
        self.evalViscousFluxZ = evalViscousFluxZLA_IP

        #self.getGs = getGsLA
        self.getGsX = getGsD_X
        self.getGsY = getGsD_Y
        self.getGsZ = getGsD_Z

      if (vflux_str == 'BR1'):
        self.evalViscousFluxX = evalViscousFluxXLA_BR1
        self.evalViscousFluxY = evalViscousFluxYLA_BR1
        self.evalTauFluxX = evalTauFluxXLA_BR1
        self.evalTauFluxY = evalTauFluxYLA_BR1
    if (check_eq == 0):
       if (mpi_rank == 0): print('Equation set ' + str(eq_str) + ' is not valid, PyDG quitting')
       sys.exit()

