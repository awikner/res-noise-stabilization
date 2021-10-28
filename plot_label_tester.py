import numpy as np
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

rhos = np.arange(0.1,1.0,0.1)
sigmas = np.ones(rhos.size)*0.5
leakages = np.ones(rhos.size)*0.125
noisevals = np.ones(rhos.size)*4.3e-3
regs = np.ones(rhos.size)*1e-5
stable_frac = 1.0-rhos

fig, (ax1, ax2) = plt.subplots(2,1, sharey = True, figsize = (14,8))
ax1.plot(rhos, stable_frac, '-bx')
ax1.set_ylabel('Stable Fraction')
ax1.set_xlabel('Spectral Radius')
for (x,y,sigma,leakage,noise,reg) in zip(rhos,stable_frac,sigmas,leakages,noisevals,regs):
    label = 'Sigma: %0.1f' % sigma + '\n' + 'Leakage: %0.3f' % leakage + '\n' + 'Noise: %0.2e' % noise
    ax1.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')

plt.show()
