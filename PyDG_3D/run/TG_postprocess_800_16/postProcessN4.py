from pylab import *
axis_font = {'size':'24'}
p3 = load('p4.npz')
p3_MZ = load('p4_MZ.npz')
p7F = load('p8F.npz')

#DNS = load('PySpec_400_128.npz')

figure(1)
plot(p3['t'][:],p3['E'],lw=1.5,color='blue',label=r'$p=3$')
plot(p3_MZ['t'][:],p3_MZ['E'],lw=1.5,color='blue',ls='--',label=r'$p=3 MZ$')
plot(p7F['t'][:],p7F['E'],lw=1.5,color='black',ls='-',label=r'$p=15$ Filtered')

#plot(DNS['t'][:],DNS['Energy'],lw=.75,color='black',label='DNS')
ylim([0.04,0.14])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$E$',**axis_font)
legend(loc=1)
savefig('TGV32_Energy.png')

figure(2)
plot(p3['t'][1::],-diff(p3['E'])*1./p3['t'][1],lw=1.5,color='blue',label=r'$p=3$')
plot(p3_MZ['t'][1::],-diff(p3_MZ['E'])*1./p3_MZ['t'][1],lw=1.5,color='blue',ls='--',label=r'$p=3MZ$')
plot(p7F['t'][1::],-diff(p7F['E'])*1./p7F['t'][1],lw=1.5,color='black',ls='-',label=r'$p=15$ Filtered')

#plot(DNS['t'][1::],-DNS['Dissipation_resolved'],lw=.75,color='black',label='DNS')
ylim([0,0.02])
xlim([0,10])
xlabel(r'$t$',**axis_font)
ylabel(r'$-\frac{dE}{dt}$',**axis_font)
legend(loc=2)
savefig('TGV32_Dissipation.png')


#data = np.load('npspec500.npz')
figure(3)
loglog(p3['k'],0.5*sum(p3['spectrum'],axis=1),color='blue',lw=1.5,label=r'$p=3$')
loglog(p3_MZ['k'],0.5*sum(p3_MZ['spectrum'],axis=1),color='blue',ls='--',lw=1.5,label=r'$p=3 MZ$')

loglog(p7F['k'],0.5*sum(p7F['spectrum'],axis=1),color='black',ls='-',lw=1.5,label=r'$p=15$ Filtered')

#loglog(data['k'],0.5*(data['E'][:,0] + data['E'][:,1] + data['E'][:,2]),'-',color='black',lw=0.75,label=r'Spectral')
xlim([2,16])
#xlim([2,amax(data['k']) + 10])
ylim([1e-4,0.05])
xlabel(r'$k$',**axis_font)
ylabel(r'$E(k)$',**axis_font)
legend(loc=1)
savefig('TGV32_Spectrum.png')

show()
