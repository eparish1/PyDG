import numpy as np
from chemistry_values import *
import cantera as ct
def add_reacting_to_main(main,mol_str):
  main.reacting = True
  main.nspecies = np.size(mol_str)
  if (main.mpi_rank == 0):
    sys.stdout.write('Chemistry Set: \n')
    for i in range(0,main.nspecies):
      sys.stdout.write(mol_str[i] + '\n')
    sys.stdout.write('============== \n')
  main.mol_str = mol_str
  main.W = getMolecularWeight(main.mol_str)
  main.D_Vols = getDiffusionVolumes(main.mol_str)
  main.delta_h0 = getEnthalpyOfFormation(main.mol_str)
  main.nasa_coeffs = getNASAPolys(mol_str)
  main.Cv = getConstantCvs(mol_str)
  main.Cp = getConstantCps(mol_str)
  main.cgas = ct.Solution('2S_CH4_BFER.cti')
  main.cgas_field = ct.SolutionArray(main.cgas,(np.size(main.a.u[0])))
  main.cgas_field_dummy = ct.SolutionArray(main.cgas,(np.size(main.a.u[0,:,:,:,0,:,:,:,0])))

  main.cgas_field_LR = ct.SolutionArray(main.cgas,(np.size(main.a.uL[0,:,:,:,0:-1,:,:])))
  main.cgas_field_L = ct.SolutionArray(main.cgas,(np.size(main.a.uR[0])))
  main.cgas_field_R = ct.SolutionArray(main.cgas,(np.size(main.a.uR[0])))
  main.cgas_field_UD = ct.SolutionArray(main.cgas,(np.size(main.a.uU[0,:,:,:,:,0:-1,:])))
  main.cgas_field_U = ct.SolutionArray(main.cgas,(np.size(main.a.uD[0])))
  main.cgas_field_D = ct.SolutionArray(main.cgas,(np.size(main.a.uD[0])))
  main.cgas_field_FB = ct.SolutionArray(main.cgas,(np.size(main.a.uF[0,:,:,:,:,:,0:-1])))
  main.cgas_field_F = ct.SolutionArray(main.cgas,(np.size(main.a.uB[0])))
  main.cgas_field_B = ct.SolutionArray(main.cgas,(np.size(main.a.uB[0])))


  main.cgas_field_L_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uL_edge[0])))
  main.cgas_field_R_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uR_edge[0])))
  main.cgas_field_D_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uD_edge[0])))
  main.cgas_field_U_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uD_edge[0])))
  main.cgas_field_B_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uF_edge[0])))
  main.cgas_field_F_edge = ct.SolutionArray(main.cgas,(np.size(main.a.uF_edge[0])))
  main.a.p = np.zeros(np.shape(main.a.u[0]))
  main.a.pR = np.zeros(np.shape(main.a.uR[0]))
  main.a.pL = np.zeros(np.shape(main.a.uL[0]))
  main.a.pU = np.zeros(np.shape(main.a.uU[0]))
  main.a.pD = np.zeros(np.shape(main.a.uD[0]))
  main.a.pF = np.zeros(np.shape(main.a.uF[0]))
  main.a.pB = np.zeros(np.shape(main.a.uB[0]))
  main.a.pR_edge = np.zeros(np.shape(main.a.uR_edge[0]))
  main.a.pL_edge = np.zeros(np.shape(main.a.uL_edge[0]))
  main.a.pU_edge = np.zeros(np.shape(main.a.uU_edge[0]))
  main.a.pD_edge = np.zeros(np.shape(main.a.uD_edge[0]))
  main.a.pF_edge = np.zeros(np.shape(main.a.uF_edge[0]))
  main.a.pB_edge = np.zeros(np.shape(main.a.uB_edge[0]))

  main.a.T = np.zeros(np.shape(main.a.u[0]))
  main.a.TR = np.zeros(np.shape(main.a.uR[0]))
  main.a.TL = np.zeros(np.shape(main.a.uL[0]))
  main.a.TU = np.zeros(np.shape(main.a.uU[0]))
  main.a.TD = np.zeros(np.shape(main.a.uD[0]))
  main.a.TF = np.zeros(np.shape(main.a.uF[0]))
  main.a.TB = np.zeros(np.shape(main.a.uB[0]))
  main.a.TR_edge = np.zeros(np.shape(main.a.uR_edge[0]))
  main.a.TL_edge = np.zeros(np.shape(main.a.uL_edge[0]))
  main.a.TU_edge = np.zeros(np.shape(main.a.uU_edge[0]))
  main.a.TD_edge = np.zeros(np.shape(main.a.uD_edge[0]))
  main.a.TF_edge = np.zeros(np.shape(main.a.uF_edge[0]))
  main.a.TB_edge = np.zeros(np.shape(main.a.uB_edge[0]))

  return main
