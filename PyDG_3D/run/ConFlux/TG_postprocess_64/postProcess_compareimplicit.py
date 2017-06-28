from pylab import *
axis_font = {'size':'24'}
p7 = load('p78.npz')
#p32 = load('p321.npz')
p32 = load('p78_implicit.npz')


figure(1)
plot(p7['t'][:],p7['E'],lw=1.5,color='green',label=r'$p=7$')
plot(p32['t'][:],p32['E'],lw=1.5,ls='--',color='red',label=r'$p=7$ Implicit')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV64_Energy.png')

figure(2)
plot(p7['t'][1::],-diff(p7['E'])*1./p7['t'][1],lw=1.5,color='green',label=r'$p=7$')
plot(p32['t'][1::],-diff(p32['E'])*1./p32['t'][1],ls='--',lw=1.5,color='red',label=r'$p=7$ Implicit')

ylim([0,0.014])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=2)


figure(3)
loglog(p7['k'],0.5*sum(p7['spectrum'],axis=1),color='green',lw=1.5,label=r'$p=7$')
loglog(p32['k'],0.5*sum(p32['spectrum'],axis=1),color='red',ls='--',lw=1.5,label=r'$p=7$ Implicit')

xlim([2,amax(p32['k']) + 10])
ylim([1e-8,0.1])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)

show()
