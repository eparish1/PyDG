import numpy as np
import sys
def computeDAB_Fuller(W1,W2,p,T,Vol1_sum,Vol2_sum):
  # computes the diffusion coefficient between two species using Fuller correlation
  # W = Molecular weight, p = pressure (atm), T = Temp (K), and Vol_sum is the tabulated diffusion volume
  Dab = 10.e-3*T**(1.75)*(1./W1 + 1./W2)**(0.5)  / ( p/101325.* (Vol1_sum**(1./3.) + Vol2_sum**(1./3.) )**2.)
  Dab *= 1e-4 #convert from cm^2/s to m^2/s
  return Dab


def getMolecularWeight(mol_str):
  # returns the molecular weight in kg / mol
  mol_weight = np.zeros(np.size(mol_str))
  H = 1.0079
  He = 4.0026
  C = 12.0107
  N = 14.0067
  O = 15.9994
  Ne = 20.1797
  Na = 22.9897
  Air = 28.97
  H2 = 2.*H
  H2O = 2*H + O
  CO = C + O
  CH = C + H
  CH4 = C + 4.*H
  CO2 = C + O*2
  S = 32.065
  CS2 = C + S*2.
  N2 = N*2.
  O2 = O*2
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      mol_weight[i] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated. Change molecules or add molecular weight to database \n')
      sys.exit()
  return mol_weight/1000.

def getDiffusionVolumes(mol_str):
  mol_volumes = np.zeros(np.size(mol_str))
  C = 16.5
  H = 1.98
  O = 5.48
  N = 5.69
  Cl = 19.5
  S = 17.0
  H2 = 7.07
  D2 = 6.70
  He = 2.88
  N2 = 17.9
  O2 = 16.6
  Air = 20.1
  Ne = 5.59
  Ar = 16.1
  Kr = 22.8
  Xe = 37.9
  CO = 18.9
  CO2 = 26.9
  N2O = 35.9
  NH3 = 14.9
  H2O = 12.7
  CCl2F2 = 114.8
  SF6 = 69.7
  Cl2 = 37.7
  BR2 = 67.2
  SO2 = 41.1
  CH4 = C + 4*H
  CS2 = C + 2*S
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      mol_volumes[i] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated in diffusivity_values.py. Change molecules or add diffusion volume for molecule to diffusivity_values.py \n')
      sys.exit()
  return mol_volumes 

def getEnthalpyOfFormation(mol_str):
  delta_h0 = np.zeros(np.size(mol_str))
  # enthalpies of formation in J/kg
  CH4 = -4675. 
  C3H8 = -2360.
  C8H18 = -1829.
  CO2 = -8943.
  H2O = -13435.
  CO = -1110.5
  O2 = 0.
  H2 = 0.
  He = 0.
  N2 = 0.
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      delta_h0[i] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated in getEnthalpyOfFormation. Change molecules or add enthalpy of formation \n')
      sys.exit()
  return delta_h0*1000. 

def getNASAPolys(mol_str):
  #Oxygen
  coeffs = np.zeros((np.size(mol_str),14))
  O = np.array([2.56942078E+00, -8.59741137E-05, 4.19484589E-08, -1.00177799E-11, 1.22833691E-15,\
                2.92175791E+04, 4.78433864E+00, 3.16826710E+00, -3.27931884E-03,  6.64306396E-06,\
                -6.12806624E-09, 2.11265971E-12, 2.91222592E+04, 2.05193346E+00])                   

  O2 = np.array([3.28253784E+00, 1.48308754E-03, -7.57966669E-07, 2.09470555E-10, -2.16717794E-14,\
                 -1.08845772E+03, 5.45323129E+00, 3.78245636E+00, -2.99673416E-03, 9.84730201E-06,\
                 -9.68129509E-09, 3.24372837E-12, -1.06394356E+03, 3.65767573E+00]) 

  H = np.array([2.50000001E+00, -2.30842973E-11, 1.61561948E-14, -4.73515235E-18, 4.98197357E-22,\
                2.54736599E+04, -4.46682914E-01, 2.50000000E+00, 7.05332819E-13, -1.99591964E-15,\
                2.30081632E-18, -9.27732332E-22, 2.54736599E+04, -4.46682853E-01])

  ### This is not correct for He. Can't find this one anywhere
  He = np.array([2.50000001E+00, -2.30842973E-11, 1.61561948E-14, -4.73515235E-18, 4.98197357E-22,\
                2.54736599E+04, -4.46682914E-01, 2.50000000E+00, 7.05332819E-13, -1.99591964E-15,\
                2.30081632E-18, -9.27732332E-22, 2.54736599E+04, -4.46682853E-01])

  H2 = np.array([3.33727920E+00, -4.94024731E-05, 4.99456778E-07, -1.79566394E-10, 2.00255376E-14,\
                 -9.50158922E+02, -3.20502331E+00, 2.34433112E+00, 7.98052075E-03, -1.94781510E-05,\
                 2.01572094E-08, -7.37611761E-12, -9.17935173E+02, 6.83010238E-01])

  OH = np.array([3.09288767E+00, 5.48429716E-04, 1.26505228E-07, -8.79461556E-11, 1.17412376E-14,\
                 3.85865700E+03, 4.47669610E+00, 3.99201543E+00, -2.40131752E-03, 4.61793841E-06,\
                 -3.88113333E-09, 1.36411470E-12, 3.61508056E+03, -1.03925458E-01])

  H2O = np.array([3.03399249E+00, 2.17691804E-03, -1.64072518E-07, -9.70419870E-11, 1.68200992E-14,\
                  -3.00042971E+04, 4.96677010E+00, 4.19864056E+00, -2.03643410E-03, 6.52040211E-06,\
                  -5.48797062E-09, 1.77197817E-12,-3.02937267E+04,-8.49032208E-01])                

  CH4 = np.array([7.48514950E-02, 1.33909467E-02, -5.73285809E-06, 1.22292535E-09, -1.01815230E-13,\
                  -9.46834459E+03, 1.84373180E+01, 5.14987613E+00, -1.36709788E-02, 4.91800599E-05,\
                  -4.84743026E-08, 1.66693956E-11, -1.02466476E+04, -4.64130376E+00])

  CO = np.array([2.71518561E+00, 2.06252743E-03, -9.98825771E-07, 2.30053008E-10, -2.03647716E-14,\
                -1.41518724E+04, 7.81868772E+00,  3.57953347E+00, -6.10353680E-04, 1.01681433E-06,\
                 9.07005884E-10, -9.04424499E-13, -1.43440860E+04, 3.50840928E+00])

  CO2 = np.array([3.85746029E+00, 4.41437026E-03, -2.21481404E-06, 5.23490188E-10, -4.72084164E-14,\
                 -4.87591660E+04, 2.27163806E+00, 2.35677352E+00, 8.98459677E-03, -7.12356269E-06,\
                  2.45919022E-09, -1.43699548E-13, -4.83719697E+04, 9.90105222E+00])

  N2 = np.array([0.02926640E+02, 0.14879768E-02, -0.05684760E-05, 0.10097038E-09, -0.06753351E-13,\
                 -0.09227977E+04, 0.05980528E+02, 0.03298677E+02, 0.14082404E-02, -0.03963222E-04,\
                  0.05641515E-07, -0.02444854E-10, -0.10208999E+04, 0.03950372E+02])
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      coeffs[i,:] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated in getNASAPolys. Change molecules or add polynomial coefficients \n')
      sys.exit()
  return coeffs 


def getConstantCvs(mol_str):
  #return Cv in J/(kg K)
  Cv = np.zeros(np.size(mol_str))
  Air = 0.718
  O2 = 0.659
  CH4 = 1.70
  CO = 0.72
  CO2 = 0.655
  H2O = 1.46
  N2 = 0.743
  H2 = 10.06
  He = 3.1156
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      Cv[i] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated in getConstantCvs. Change molecules or add Cvs \n')
      sys.exit()
  return Cv*1000.



def getConstantCps(mol_str):
  # returns Cp in J/(kg K)
  Cp = np.zeros(np.size(mol_str))
  Air = 1.01
  O2 = 0.919
  CH4 = 2.22
  CO = 1.02
  CO2 = 0.844
  H2O = 1.93
  N2 = 1.04
  H2 = 14.32
  He = 5.1926
  for i in range(0,np.size(mol_str)):
    if mol_str[i] in locals():
      Cp[i] = locals()[mol_str[i]]
    else:
      sys.stdout.write('Error, molecule ' + mol_str[i] + ' not tabluated in getConstantCps. Change molecules or add Cps \n')
      sys.exit()
  return Cp*1000.

