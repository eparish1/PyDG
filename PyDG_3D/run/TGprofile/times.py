from pylab import *
close("all")

axis_font = {'size':'24'}
DOFs = 64**3
Nel = np.array([64,32,16,8,4,2])
p = np.array([0,1,3,7,15,31])
CFL = 1./(Nel*(2.*p+1))
times = 1./(10.*4.*DOFs)*np.array([200.135917902-3.67774486542,122.432684898-2.66475582123,90.916711092-2.82900309563,117.162789106-34.8705589771,92.6268517971,154.272378922])


plot(p[:],times[:]*1e6,'-o',lw=1.5,color='blue')
xlabel(r'$p$',**axis_font)
ylabel(r'CPU time per timestep ($\mu s$)',**axis_font)
savefig('pvsh_cputime.png')

## Scaling (local machine)
figure(2)
nproc = np.array([1,2,4,8,16,32,64])
ptimes = np.array([216.906772852,134.440196991,75.5704569817,40.084580183, 28.869243145,18.6457509995,12.931153059])
ptimes_ideal = ptimes[0]/nproc
loglog(nproc,ptimes,'-o',color='blue',lw=1.5)
loglog(nproc,ptimes_ideal,color='blue',ls='--',lw=0.75)
xlabel('Number of cores',**axis_font)
ylabel(r'CPU time',**axis_font)


figure(3)
plot(p[:],times[:]*1e6/CFL,'-o',lw=1.5,color='blue')
xlabel(r'$p$',**axis_font)
ylabel(r'CPU time per timestep/CFL$^*$($\mu s$)',**axis_font)
savefig('CFL_pvsh_cputime.png')
show()

