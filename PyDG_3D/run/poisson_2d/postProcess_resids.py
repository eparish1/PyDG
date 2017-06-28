from pylab import *
axis_font = {'size':'24'}
p4_n8 = load('resid_p4_n8.npz')
p4_n8_MG = load('resid_p4_n8_MG.npz')

p4_n16 = load('resid_p4_n16.npz')
p4_n16_MG = load('resid_p4_n16_MG.npz')

p4_n32 = load('resid_p4_n32.npz')
p4_n32_MG = load('resid_p4_n32_MG.npz')

loglog(p4_n8['t'],p4_n8['resid'],ls='-',lw=1.5,color='blue',label=r'$p=4,N=8$')
loglog(p4_n16['t'],p4_n16['resid'],ls='-',color='red',lw=1.5,label=r'$p=4,N=16$')
loglog(p4_n32['t'],p4_n32['resid'],ls='-',color='green',lw=1.5,label=r'$p=4,N=32$')

loglog(p4_n8_MG['t'],p4_n8_MG['resid'],ls='--',color='blue',lw=1.5,label=r'MG, $p=4,N=8$')
loglog(p4_n16_MG['t'],p4_n16_MG['resid'],ls='--',color='red',lw=1.5,label=r'MG, $p=4,N=16$')
loglog(p4_n32_MG['t'],p4_n32_MG['resid'],ls='--',color='green',lw=1.5,label=r'MG, $p=4,N=32$')

xlim([0.1,10**4])
legend(loc=1)
xlabel(r'CPU time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
savefig('poisson_mg_p4.pdf')


figure(2)
p8_n4 = load('resid_p8_n4.npz')
p8_n4_MG = load('resid_p8_n4_MG.npz')


p8_n8 = load('resid_p8_n8.npz')
p8_n8_MG = load('resid_p8_n8_MG.npz')

p8_n16 = load('resid_p8_n16.npz')
p8_n16_MG = load('resid_p8_n16_MG.npz')


loglog(p8_n4['t'],p8_n4['resid'],ls='-',color='blue',lw=1.5,label=r'$p=8,N=4$')
loglog(p8_n8['t'],p8_n8['resid'],ls='-',color='red',lw=1.5,label=r'$p=8,N=8$')
loglog(p8_n16['t'],p8_n16['resid'],ls='-',color='green',lw=1.5,label=r'$p=8,N=16$')

loglog(p8_n4_MG['t'],p8_n4_MG['resid'],ls='--',color='blue',lw=1.5,label=r'MG, $p=8,N=4$')
loglog(p8_n8_MG['t'],p8_n8_MG['resid'],ls='--',color='red',lw=1.5,label=r'MG, $p=8,N=8$')
loglog(p8_n16_MG['t'],p8_n16_MG['resid'],ls='--',color='green',lw=1.5,label=r'MG, $p=8,N=16$')

xlim([0.1,10**4])
legend(loc=1)
xlabel(r'CPU time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
savefig('poisson_mg_p8.pdf')


figure(3)
p16_n2 = load('resid_p16_n2.npz')
p16_n2_MG = load('resid_p16_n2_MG.npz')
p16_n4 = load('resid_p16_n4.npz')
p16_n4_MG = load('resid_p16_n4_MG.npz')
p16_n8 = load('resid_p16_n8.npz')
p16_n8_MG = load('resid_p16_n8_MG.npz')


loglog(p16_n2['t'],p16_n2['resid'],ls='-',color='blue',lw=1.5,label=r'$p=16,N=2$')
loglog(p16_n4['t'],p16_n4['resid'],ls='-',color='red',lw=1.5,label=r'$p=16,N=4$')
loglog(p16_n8['t'],p16_n8['resid'],ls='-',color='green',lw=1.5,label=r'$p=16,N=8$')

loglog(p16_n2_MG['t'],p16_n2_MG['resid'],ls='--',color='blue',lw=1.5,label=r'MG, $p=16,N=2$')
loglog(p16_n4_MG['t'],p16_n4_MG['resid'],ls='--',color='red',lw=1.5,label=r'MG, $p=16,N=4$')
loglog(p16_n8_MG['t'],p16_n8_MG['resid'],ls='--',color='green',lw=1.5,label=r'MG, $p=16,N=8$')

xlim([0.1,10**4])
legend(loc=1)
xlabel(r'CPU time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
savefig('poisson_mg_p16.pdf')

p32_n4_MG = load('resid_p32_n4_MG.npz')
#p64_n2_MG = load('resid_p64_n2_MG.npz')

figure(4)
loglog(p4_n32_MG['t'],p4_n32_MG['resid'],ls='--',color='black',lw=1.5,label=r'MG, $p=4,N=32$')
loglog(p8_n16_MG['t'],p8_n16_MG['resid'],ls='--',color='blue',lw=1.5,label=r'MG, $p=8,N=16$')
loglog(p16_n8_MG['t'],p16_n8_MG['resid'],ls='--',color='red',lw=1.5,label=r'MG, $p=16,N=8$')
loglog(p32_n4_MG['t'],p32_n4_MG['resid'],ls='--',color='green',lw=1.5,label=r'MG, $p=32,N=4$')
#loglog(p64_n2_MG['t'],p64_n2_MG['resid'],ls='--',color='orange',lw=1.5,label=r'MG, $p=64,N=2$')

xlim([0.1,10**4])
legend(loc=1)
xlabel(r'CPU time (s)',**axis_font)
ylabel(r'Residual',**axis_font)
savefig('poisson_mg_ps.pdf')
show()
