import numpy as np
from pqr2grid import GriddedPQR

pqrfile = 'pqr/clya_rr.pqr'
outname = 'clya_rr_pH7.0'


r = np.arange(0,7.05,0.05)
z = np.arange(-3.5,13.55,0.05)


cm = GriddedPQR(pqrfile)

cm.load_pqr()
cm.compute_chargemap(r, z, sigma=0.5, selection='all')
cm.chargemap.write_comsol_griddata(
    f'{outname:s}_chargemap.txt', normalization=1e-9
)