import numpy as np
from chemistry_values import *
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
  return main
