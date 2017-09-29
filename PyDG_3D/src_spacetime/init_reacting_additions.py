import numpy as np
from chemistry_values import *
def add_reacting_to_main(main,mol_str):
  main.reacting = True
  main.nspecies = np.size(mol_str)
  if (main.mpi_rank == 0 and main.t == 0):
    sys.stdout.write('Chemistry Set: \n')
    for i in range(0,main.nspecies):
      sys.stdout.write(mol_str[i] + '\n')
    sys.stdout.write('============== \n')
  main.mol_str = mol_str
  main.W = getMolecularWeight(main.mol_str)
  main.D_Vols = getDiffusionVolumes(main.mol_str)
  main.delta_h0 = getEnthalpyOfFormation(main.mol_str)
  main.nasa_coeffs = getNASAPolys(mol_str)
  #main.Cv = getConstantCvs(mol_str)
  main.Cp = getConstantCps(mol_str)
  #main.Cp[:] = main.Cp[0]
  #main.Cv[:] = main.Cv[0]
  #main.delta_h0[:] = main.delta_h0[0]
  #main.W[:] = main.W[0]
  main.gamma = main.Cp/main.Cv
  #main.gamma[:] = 1.4
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

  main.a.rh0 = np.zeros(np.shape(main.a.u[0]))
  main.a.gamma_star = np.zeros(np.shape(main.a.u[0]))

  return main
