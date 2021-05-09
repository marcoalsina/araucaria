from numpy import arange, sin, pi
from scipy.fftpack import fftfreq
from araucaria.xas import ftwindow, xftf_kwin, xftr_kwin
nfft = 2048  # number of points for FFT
ks   = 0.05  # delta k (angstrom^-1)
f1   = 0.5   # freq1 (angstrom)
k    = arange(0, 10, ks)
wink = ftwindow(k, x_range=(0,10), dx1=0.5, win='sine')
chi  = 0.5*sin(2*pi*k*f1)
chir = xftf_kwin(wink*chi, nfft=nfft, kstep=ks)
freq = fftfreq(nfft, ks)[:nfft//2]
chiq = xftr_kwin(chir, nfft=nfft, kstep=ks)[:len(k)]
print(chiq.dtype)
# complex128

# plotting reverse FFT signal
import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
fig, ax = fig_xas_template(panels='re', fig_pars={'kweight':0})
line = ax[0].plot(freq, abs(chir))
xlim = ax[0].set_xlim(0,2)
xlab = ax[0].set_xlabel('$R/\pi$ [$\AA$]')
line = ax[1].plot(k, chiq)
text = ax[1].set_xlabel(r'$q(\AA^{-1})$')
text = ax[1].set_ylabel(r'$\chi(q)$')
fig.tight_layout()
plt.show(block=False)
