from pylab import *
axis_font = {'size':'24'}
pe1_p4_n16 = load('pe1_p4_n16.npz')
pe1_p4_n16_MG = load('pe1_p4_n16_mg.npz')

pe1_p4_n32 = load('pe1_p4_n32.npz')
pe1_p4_n32_MG = load('pe1_p4_n32_mg.npz')

#pe1_p4_n64 = load('pe1_p4_n64.npz')
#pe1_p4_n64_MG = load('pe1_p4_n64.npz')

loglog(pe1_p4_n16['t'],pe1_p4_n16['resid'],ls='-',lw=1.5,color='blue',label='GMRES, $N=16$')
loglog(pe1_p4_n32['t'],pe1_p4_n32['resid'],ls='-',lw=1.5,color='red',label='GMRES, $N=32$')

loglog(pe1_p4_n16_MG['t'],pe1_p4_n16_MG['resid'],ls='--',lw=1.5,color='blue',label='MG, $N=16$')
loglog(pe1_p4_n32_MG['t'],pe1_p4_n32_MG['resid'],ls='--',lw=1.5,color='red',label='MG, $N=32$')

xlabel(r'CPU Time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
legend(loc=1)
#plot(pe1_p4_n64['t'],pe1_p4_n64['resid'],ls='-',lw=1.5,color='green',label='GMRES, $N=64$')
#plot(pe1_p4_n64_MG['t'],pe1_p4_n64_MG['resid'],ls='--',lw=1.5,color='green',label='MG, $N=64$')


figure(2)
pe10_p4_n16 = load('pe10_p4_n16.npz')
pe10_p4_n16_MG = load('pe10_p4_n16_mg.npz')

pe10_p4_n32 = load('pe10_p4_n32.npz')
pe10_p4_n32_MG = load('pe10_p4_n32_mg.npz')

#pe1_p4_n64 = load('pe1_p4_n64.npz')
#pe1_p4_n64_MG = load('pe1_p4_n64.npz')

loglog(pe10_p4_n16['t'],pe10_p4_n16['resid'],ls='-',lw=1.5,color='blue',label='GMRES, $N=16$')
loglog(pe10_p4_n32['t'],pe10_p4_n32['resid'],ls='-',lw=1.5,color='red',label='GMRES, $N=32$')

loglog(pe10_p4_n16_MG['t'],pe10_p4_n16_MG['resid'],ls='--',lw=1.5,color='blue',label='MG, $N=16$')
loglog(pe10_p4_n32_MG['t'],pe10_p4_n32_MG['resid'],ls='--',lw=1.5,color='red',label='MG, $N=32$')

xlabel(r'CPU Time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
legend(loc=1)

#plot(pe1_p4_n64['t'],pe1_p4_n64['resid'],ls='-',lw=1.5,color='green',label='GMRES, $N=64$')
#plot(pe1_p4_n64_MG['t'],pe1_p4_n64_MG['resid'],ls='--',lw=1.5,color='green',label='MG, $N=64$')

figure(3)
pe100_p4_n16 = load('pe100_p4_n16.npz')
pe100_p4_n16_MG = load('pe100_p4_n16_mg.npz')

pe100_p4_n32 = load('pe100_p4_n32.npz')
pe100_p4_n32_MG = load('pe100_p4_n32_mg.npz')

pe100_p4_n64 = load('pe100_p4_n64.npz')
pe100_p4_n64_MG = load('pe100_p4_n64_mg.npz')

loglog(pe100_p4_n16['t'],pe100_p4_n16['resid'],ls='-',lw=1.5,color='blue',label='GMRES, $N=16$')
loglog(pe100_p4_n32['t'],pe100_p4_n32['resid'],ls='-',lw=1.5,color='red',label='GMRES, $N=32$')
loglog(pe100_p4_n64['t'],pe100_p4_n64['resid'],ls='-',lw=1.5,color='green',label='GMRES, $N=64$')

loglog(pe100_p4_n16_MG['t'],pe100_p4_n16_MG['resid'],ls='--',lw=1.5,color='blue',label='MG, $N=16$')
loglog(pe100_p4_n32_MG['t'],pe100_p4_n32_MG['resid'],ls='--',lw=1.5,color='red',label='MG, $N=32$')
loglog(pe100_p4_n64_MG['t'],pe100_p4_n64_MG['resid'],ls='--',lw=1.5,color='green',label='MG, $N=64$')

xlabel(r'CPU Time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
legend(loc=1)


figure(4)
pe1000_p4_n16 = load('pe1000_p4_n16.npz')
pe1000_p4_n16_MG = load('pe1000_p4_n16_mg.npz')

pe1000_p4_n32 = load('pe1000_p4_n32.npz')
pe1000_p4_n32_MG = load('pe1000_p4_n32_mg.npz')

pe1000_p4_n64 = load('pe1000_p4_n64.npz')
pe1000_p4_n64_MG = load('pe1000_p4_n64_mg.npz')

loglog(pe1000_p4_n16['t'],pe1000_p4_n16['resid'],ls='-',lw=1.5,color='blue',label='GMRES, $N=16$')
loglog(pe1000_p4_n32['t'],pe1000_p4_n32['resid'],ls='-',lw=1.5,color='red',label='GMRES, $N=32$')
plot(pe1000_p4_n64['t'],pe1000_p4_n64['resid'],ls='-',lw=1.5,color='green',label='GMRES, $N=64$')

loglog(pe1000_p4_n16_MG['t'],pe1000_p4_n16_MG['resid'],ls='--',lw=1.5,color='blue',label='MG, $N=16$')
loglog(pe1000_p4_n32_MG['t'],pe1000_p4_n32_MG['resid'],ls='--',lw=1.5,color='red',label='MG, $N=32$')
plot(pe1000_p4_n64_MG['t'],pe1000_p4_n64_MG['resid'],ls='--',lw=1.5,color='green',label='MG, $N=64$')

xlabel(r'CPU Time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
legend(loc=1)

show()
