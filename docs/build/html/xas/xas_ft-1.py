from numpy import arange
import matplotlib.pyplot as plt
from araucaria.xas import ftwindow
from araucaria.plot import fig_xas_template
k       = arange(0, 10.1, 0.05)
k_range = [2,8]
windows = ['hanning', 'parzen', 'welch',
           'gaussian', 'sine', 'kaiser']
dk      = 1.0
fig_kws = {'sharex' : True}
fig, axes = fig_xas_template(panels='ee/ee/ee', **fig_kws)
for i, ax in enumerate(axes.flatten()):
    win  = ftwindow(k, k_range, dk, win= windows[i])
    line = ax.plot(k, win, label=windows[i])
    for val in k_range:
        line = ax.axvline(val - dk/2, color='gray', ls=':')
        line = ax.axvline(val + dk/2, color='gray', ls=':')
    leg  = ax.legend()
    text = ax.set_ylabel('')
    text = ax.set_xlabel('')
fig.tight_layout()
plt.show(block=False)
