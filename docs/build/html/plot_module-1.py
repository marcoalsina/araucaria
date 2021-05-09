import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
pars = {'e_range' : (0,100),
        'mu_range': (0,1.5),
        'k_range' : (0,15),
        'r_range' : (0,6)}
fig, axes = fig_xas_template('dx/er', fig_pars=pars)
plt.show(block=False)
