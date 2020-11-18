from numpy import arange, sin, pi
from scipy.fftpack import fftfreq
from araucaria.xas import ftwindow, xftf_kwin
nfft = 2048  # number of points for FFT
ks   = 0.05  # delta k (angstrom^-1)
f1   = 0.5   # freq1 (angstrom)
f2   = 1.2   # freq2 (angstrom)
k    = arange(0, 10, ks)
win  = ftwindow(k, x_range=(0,10), dx1=0.5, win='sine')
chi  = 0.5*sin(2*pi*k*f1) + 0.1*sin(2*pi*k*f2)
chir = xftf_kwin(win*chi, nfft=nfft, kstep=ks)
freq = fftfreq(nfft, ks)
print(chir.dtype)
# complex128

# plotting forward FFT signal
import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
fig, ax = fig_xas_template(panels='er', fig_pars={'kweight':0})
line = ax[0].plot(k, win*chi)
line = ax[1].plot(freq[:int(nfft/2)], abs(chir[:int(nfft/2)]))
xlim = ax[1].set_xlim(0,2)
xlab = ax[1].set_xlabel('$R/\pi$ [$\AA$]')
for f in (f1,f2):
    line = ax[1].axvline(f, color='gray', ls=':')
fig.tight_layout()
plt.show(block=False)
