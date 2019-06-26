"""Grid module


"""

from abc import ABCMeta
from typing import Literal

from numpy import (
    ndarray,
    meshgrid,
    zeros,
    savetxt,
    where,
    hypot,
    array,
    histogram,
    arange
)
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

from skimage import measure
from shapely.geometry import Polygon


class Grid(metaclass=ABCMeta):
    """
    Class representing a grid of data.
    """

    def __init__(
        self,
        vectors: list[ndarray],
        initial_values: float = 0
    ):
        self.vectors: list[ndarray] = vectors
        self.meshgrid: list[ndarray] = meshgrid(*self.vectors)
        self.data: ndarray = zeros(self.meshgrid[0].shape) + initial_values

    @property
    def shape(
        self
    ) -> tuple:
        return self.data.shape

    @property
    def dims(
        self
    ) -> int:
        return len(self.shape)

    @property
    def mincoords(
        self
    ) -> list[float]:
        return [v[0] for v in self.vectors]

    @property
    def maxcoords(
        self
    ) -> list[float]:
        return [v[-1] for v in self.vectors]

    @property
    def spacing(
        self
    ) -> list[float]:
        return [v[1] - v[0] for v in self.vectors]

    def write_comsol_griddata(
        self,
        filename: str,
        normalization: float = 1
    ) -> None:
        """
        Write out the grid to a COMSOL grid data format file.

        Parameters
        ----------
        filename : str
            Name of the file to save.
        normalization : float, optional
            Dimensional normalization, by default 1. All vectors will be
            multiplied by this value, whereas the grid data will be divided by
            normalization to the power of grid dimensionality.
        """

        # Write header
        with open(filename, "w") as fid:
            print("%Grid", file=fid)

        with open(filename, "a") as fid:
            # Write vectors
            for vector in self.vectors:
                savetxt(fid, vector.T * normalization,
                        fmt="%.10e", delimiter=",", newline=",")
                print("", file=fid)
            print("\r", file=fid)
            # Write data
            print("\n%Data", file=fid)
            savetxt(fid, self.data * (normalization ** (-self.dims)),
                    fmt="%.10e", delimiter=",")


class Grid2D(Grid):
    """
    Class representing a grid of 2D data.
    """

    def __init__(
        self,
        x: ndarray,
        y: ndarray,
        initial_values: float = 0
    ):
        super().__init__(
           vectors=[x,y],
           initial_values=initial_values
        )

    @property
    def x(
       self
    ) -> ndarray:
        return self.vectors[0]

    @property
    def y(
       self
    ) -> ndarray:
        return self.vectors[1]

    @property
    def X(
       self
    ) -> ndarray:
        return self.meshgrid[0]

    @property
    def Y(
       self
    ) -> ndarray:
        return self.meshgrid[1]

    def create_polygon_contour(
        self,
        cutoff: float,
        simplification_factor: float = 0.1,
        y_offset: float = 0,
        filename: str | None = None
    ) -> ndarray:
        """
        Create a polygon of a contourline at the given contour cutoff value.

        Parameters
        ----------
        cutoff : float
            Contour cutoff value.
        simplification_factor : float, optional
            Factor with which to simplify the polygon, by default 0.1. Will be
            passed to `Polygon.simplify()`.
        y_offset : float, optional
            Offset distance value in the y-direction of the grid, by default 0.
            Can be used to align along the non-radial axis.
        filename : str | None, optional
            Name of the file to save the polygon to, or None to not to, by
            default None.
        """
        # Create contourline
        contours = measure.find_contours(self.data, cutoff)
        # Extract contour coordinates
        polyx = contours[0][:, 1]*self.spacing[0]
        polyy = contours[0][:, 0]*self.spacing[1] + self.mincoords[1]
        # Create polygon, simplify and apply offset
        poly = Polygon(array([polyx, polyy]).T)
        coords = array(list(
            poly.simplify(simplification_factor).exterior.coords
        ))
        coords[:,1] = coords[:,1] + y_offset
        # Save coordinates to file
        if filename:
            savetxt(filename, coords, fmt='%.2f', delimiter=' ', newline='\n')

        return coords


class Grid3D(Grid):
    """
    Class representing a grid of 3D data.
    """

    def __init__(
        self, 
        x: ndarray,
        y: ndarray,
        z: ndarray,
        initial_values: float = 0
    ):
        super().__init__(
           vectors=[x,y,z],
           initial_values=initial_values
        )

    @property
    def x(
       self
    ) -> ndarray:
        return self.vectors[0]

    @property
    def y(
       self
    ) -> ndarray:
        return self.vectors[1]

    @property
    def z(
       self
    ) -> ndarray:
        return self.vectors[2]

    @property
    def X(
       self
    ) -> ndarray:
        return self.meshgrid[0]

    @property
    def Y(
       self
    ) -> ndarray:
        return self.meshgrid[1]

    @property
    def Z(
       self
    ) -> ndarray:
        return self.meshgrid[2]

    def average_radially(
        self,
        center: tuple[float, float] = (0,0)
    ) -> Grid2D:
        """
        Average the grid radially, alond the z-axis.

        Parameters
        ----------
        center : tuple, optional
            The x-, and y-coordinates of the grid center, by default (0,0).

        Returns
        -------
        Grid2D
            Radially averaged (i.e., 2D-axisymmetric approximation) version of
            this 3D grid.
        """

        # Create bins
        bins = arange(
            0,
            self.X.max(),
            (self.X.max() - self.X.min() + 1) / self.X.shape[1]
        )

        # Perform radial averaging
        V, xv, yv = radial_average_kernel(
            self.data, self.X, self.Y, self.Z,
            center=center, bins=bins
        )
        # Create grid
        grid = Grid2D(x=xv, y=yv, initial_values=V)

        return grid

    def plot_slice(
        self,
        plane: Literal["yz", "xz", "xy"] = "yz",
        at=0,
        ax: Axes | None = None,
        **kwargs: dict
    ) -> QuadMesh:
        """
        Plot a slice through a grid.

        Parameters
        ----------
        plane : "yz" | "xz" | "xy", optional
             Plane to slice, by default "yz".
        at : int, optional
            Index to slice through, by default 0.
        ax : _type_, optional
             Axes object to plot in, by default None
        **kwargs : dict
            Keywords arguments passed to matplotlibs `pcolormesh` method.

        Returns
        -------
        QuadMesh
            Pcolormesh graph of the requested slice.

        Raises
        ------
        ValueError
            If `plane` is not 'yz' or 'xz' or 'xy'.
        """

        if plane == 'yz':
            ati = where(abs(self.Y[:,0,0] - at) < 1e-6)[0][0]
            Xp = self.X[ati,:,:] 
            Yp = self.Z[ati,:,:]
            data = self.data[ati,:,:]
        elif plane == 'xz':
            ati = where(abs(self.X[0,:,0] - at) < 1e-6)[0][0]
            Xp = self.Y[:,ati,:]
            Yp = self.Z[:,ati,:]
            data = self.data[:,ati,:]
        elif plane == 'xy':
            ati = where(abs(self.Z[0,0,:] - at) < 1e-6)[0][0]
            Xp = self.X[:,:,ati]
            Yp = self.Y[:,:,ati]
            data = self.data[:,:,ati]
        else:
            raise ValueError('Unsupported slice plane: {}'.format(plane))   

        if ax == None:
            ax = pyplot.gca()

        pc = ax.pcolormesh(Xp, Yp, data, **kwargs)
        return pc


#@njit(parallel=True)
def radial_average_kernel(
    V: ndarray,
    X: ndarray,
    Y: ndarray,
    Z: ndarray,
    center: tuple[float, float],
    bins: ndarray
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Kernel function for computing the radial average along the z-axis.

    Parameters
    ----------
    V : ndarray
        Grid containing the data to average.
    X, Y, Z : ndarray
        Meshgrids for the x-, y-, and z-coordinates of the grid, must have the
        same shape as `V`.
    center : (float, float)
        The x-, and y-coordinates of the grid center.
    bins: ndarray
        Radial bins to sample.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Tuple containing the average data, x-vector, and y-vector.
    """

    # Compute a radius coordinate meshgrid relative to center
    R = hypot(X[:,:,0]-center[0], Y[:,:,0]-center[1])

    averages = []
    for zi in range(0, Z.shape[2]):
        data = V[:,:,zi]
        radial_counts, radii = histogram(R, bins=bins)
        radial_profile, radii = histogram(R, weights=data, bins=bins)
        radial_mean = radial_profile / radial_counts
        averages.append(radial_mean)

    V = array(averages)
    xv = radii[:-1]
    yv = Z[0,0,:]

    return (V, xv, yv)
