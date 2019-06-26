"""
Create a 2D-axisymmetric chargemap from a PQR file.
"""

import math

import numpy as np
from numpy import ndarray

from numba import jit, njit, prange

from MDAnalysis import Universe, AtomGroup

# Local package imports
from .grid import Grid2D, Grid3D

class GriddedPQR():

    def __init__(
        self,
        pqrfile: str
    ):
        self.pqrfile: str = pqrfile
        self.universe: Universe | None = None
        self.chargemap: Grid2D | None = None
        self.densitymap: Grid3D | None = None

    def load_pqr(
        self
    ) -> None:
        """Load in molecule atoms from PQR file."""
        self.universe = Universe(self.pqrfile)

    def _extract_atom_properties(
        self,
        selection: str = "all"
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Grab lists of atom properties for given selection. 

        Parameters
        ----------
        selection : str, optional
            Atom selection string (VMD syntax), by default "all".

        Returns
        -------
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                Tuple of ndarrays (xa, ya, za, ra, qa) containing atomic
                coordinates (xa, ya, za), radii (ra) and partial charges (qa).
        """
        assert isinstance(self.universe, Universe)

        sel: AtomGroup = self.universe.select_atoms(selection)

        assert sel.n_atoms > 0

        xa, ya, za = sel.positions.T / 10
        ra = sel.radii / 10
        qa = sel.charges

        return (xa, ya, za, ra, qa)

    def compute_chargemap(
        self,
        r: np.ndarray,
        z: np.ndarray,
        sigma: float = 0.5,
        selection: str = "all"
    ) -> None:
        """
        Compute radially averaged chargemap.

        Parameters
        ----------
        r : ndarray
            Vector for grid in r-direction.
        z : ndarray
            Vector for grid in z-direction.
        sigma : float, optional
            Sharpness factor (>0) for Gaussian width, by default 0.5.
        selection : str, optional
            Atom selection string (VMD syntax) which will be used in the map
            computation, by default "all".
        """
        assert sigma > 0

        # Construct grid
        self.chargemap = Grid2D(x=r, y=z, initial_values=0)

        # Grab atom properties
        xa, ya, za, radii, charges = self._extract_atom_properties(
            selection=selection)

        # Compute chargemap data
        _compute_2D_gaussian_charge_map(
            self.chargemap.data,
            self.chargemap.X, self.chargemap.Y,
            xa, ya, za, radii, charges, sigma=sigma
        )

    def compute_densitymap(
        self,
        mincoords: tuple[float, float, float],
        maxcoords: tuple[float, float, float],
        spacing: tuple[float, float, float],
        sigma: float = 0.5,
        selection: str = 'all',
    ) -> None:
        """
        Compute 3D density map.

        Parameters
        ----------
        mincoords : (float, float, float)
            Mininum x-, y-, and z- coordinates of the density grid.
        maxcoords : (float, float, float)
            Maximum x-, y-, and z- coordinates of the density grid.
        spacing : (float, float, float)
            Spacings in the x-, y-, and z-directions.
        sigma : float, optional
            Sharpness factor (>0) for Gaussian width, by default 0.5.
        selection : str, optional
            Atom selection string (VMD syntax) which will be used in the map
            computation, by default "all".
        """
        # Set up grid vectors
        x = np.arange(mincoords[0], maxcoords[0]+spacing[0], spacing[0])
        y = np.arange(mincoords[1], maxcoords[1]+spacing[1], spacing[1])
        z = np.arange(mincoords[2], maxcoords[2]+spacing[2], spacing[2])
        # Construct grid
        self.densitymap = Grid3D(x=x, y=y, z=z, initial_values=1)
        # Grab atom properties
        xa, ya, za, radii, _ = self._extract_atom_properties(selection=selection)
        # Compute density
        _compute_3D_gaussian_density_map(
            self.densitymap.data,
            x, y, z,
            xa, ya, za, radii, sigma
        )


# @njit(parallel=True)
def _compute_2D_gaussian_charge_map(
    data: ndarray,
    R: ndarray,
    Z: ndarray,
    xa: ndarray,
    ya: ndarray,
    za: ndarray,
    radii: ndarray,
    charges: ndarray,
    sigma: float
) -> None:
    """
    Base function for computing a 2D Gaussian charge map.
    
    The resulting map will be stored in `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to store the chargemap data.
    R, Z : ndarray
        Meshgrid of r- and z-coordinates, must have the same dimensions as
        `data`.
    xa, ya, za : ndarray
        The x-, y-, and z-coordinates of all atoms.
    radii : ndarray
        The radii of all atoms.
    charges : ndarray
        The partial charges of all atoms.
    sigma : float
        Sharpness factor (>0) for Gaussian width.
    """

    # Precalculate all that we can
    rcoords = np.hypot(xa, ya).astype(np.float64)
    zcoords = za.astype(np.float64)
    widths = (sigma * radii)**(-2)
    ndim = len(R.shape)  # Dimensionality of output grid
    amplitudes = charges/np.sqrt((np.pi)**ndim * (sigma * radii)**(2 * ndim))
    # Cast as float32
    R = R.astype(np.float64)
    Z = Z.astype(np.float64)
    data = data.astype(np.float64)
    # Reset all values in the given grid to zero
    data[:] = 0
    # Determine number of atoms
    natoms = len(radii)
    # Evaluate Gaussian function for all atoms in grid
    global_counter = 0
    print(" ", global_counter, "/", natoms)
    print("\r")

    for i in prange(natoms):
        # Print progress
        if global_counter % 1 == 0:
            print("\r ", global_counter+1, "/", natoms)
            print("\r\r")
        # Do calculation
        data += amplitudes[i] * \
            np.exp(-((R - rcoords[i])**2 + (Z - zcoords[i])**2) * widths[i])
        # Update progress counter
        global_counter += 1
    print("")


@njit(fastmath=True)
def gaussian_density_jit_kernel(
    xv: ndarray,
    yv: ndarray,
    zv: ndarray,
    xai: float,
    yai: float,
    zai: float,
    wai: float
) -> ndarray:
    """
    Kernel function for evaluating the Gaussian density of an atom on a grid.

    Parameters
    ----------
    xv, yv, zv : ndarray
        Grid vectors in the x-, y-, and z-directions.
    xai, yai, zai : ndarray
        The x-, y-, and z-coordinate of the atom.
    wai : float
        Weight factor of the Gaussian for the atom.

    Returns
    -------
    ndarray
        Guassian density of the atom evaluated on a grid.
    """
    return 1 - np.exp(-((xv-xai)**2 + (yv-yai)**2 + (zv-zai)**2) * wai)

@njit(parallel=True)
def _compute_3D_gaussian_density_map(
    data: ndarray,
    xv: ndarray,
    yv: ndarray,
    zv: ndarray,
    xa: ndarray,
    ya: ndarray,
    za: ndarray,
    radii: ndarray,
    sigma: float
) -> None:
    """
    Base function for computing a 3D Gaussian density map.
    
    The resulting map will be stored in data.

    Parameters
    ----------
    data : ndarray
        Array into which to store the data.
    xv, yv, zv : ndarray
        Grid vectors in the x-, y-, and z-directions.
    xa, ya, za : ndarray
        The x-, y-, and z-coordinates of all atoms.
    radii : ndarray
        The radii of all atoms.
    sigma : float
        Sharpness factor (>0) for Gaussian width.
    """

    # Precalculate inverse of Gaussian widths
    wa = ((sigma * radii)**(-2)).astype(np.float64)

    # Pre-allocate the charge grid with ones
    data[:] = 1
    data = data.astype(np.float64)

    print(data.shape[0], data.shape[1], data.shape[2])

    # Evaluate Gaussian function for all atoms in grid
    natoms = radii.size

    print('Gridding atoms ...')
    for i in range(natoms):
        K = gaussian_density_jit_kernel(xv, yv, zv, xa[i], ya[i], za[i], wa[i])
        data = data * K

        if i % 10000 == 0:
            print(' ', i+1, '/', natoms)

    data = 1 - data
