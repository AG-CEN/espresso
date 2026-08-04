# Adapted from Kloosterman Lab (https://bitbucket.org/kloostermannerflab/fklab-python-core)
# Modified by Espresso Neuro Maintainers.
# Licensed under the GNU General Public License v3. See the root LICENSE file.

import operator
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.signal

from espresso.models.ripple_event import RippleEvent


def segment_sort(segments):
    """Sort segments by start time.

    Parameters
    ----------
    segments : segment array

    Returns
    -------
    sorted segments

    """
    segments = check_segments(segments, copy=False)
    if segments.shape[0] > 1:
        idx = np.argsort(segments[:, 0])
        segments = segments[idx, :]
    return segments


def segment_has_overlap(segments) -> bool:
    """Check for overlap of segments.

    Parameters
    ----------
    segments : segment array

    Returns
    -------
    bool
        True if any of the segments overlap.

    """
    segments = check_segments(segments, copy=True)
    segments = segment_sort(segments)

    return bool(np.any(segments[1:, 0] < segments[:-1, 1]))


def segment_remove_overlap(segments, strict=True):
    """Remove overlap between segments.

    Segments that overlap are merged.

    Parameters
    ----------
    segments : segment array
    strict : bool
        Only merge two segments if the end time of the first is stricly
        larger than (and not equal to) the start time of the second segment.

    Returns
    -------
    segments without overlap

    """

    segments = check_segments(segments, copy=False)

    n = segments.shape[0]
    if n == 0:
        return segments

    segments = segment_sort(segments)

    s = segments[:1, :]

    fcn = operator.lt if strict else operator.le

    for k in range(1, n):
        if fcn(s[-1, 1], segments[k, 0]):
            s = np.concatenate([s, segments[k : k + 1, :]])
        else:
            s[-1, 1] = np.maximum(segments[k, 1], s[-1, 1])

    return s


def segment_invert(segments):
    """Invert segments.

    Constructs segments from the inter-segment intervals.

    Parameters
    ----------
    segments : segment array

    Returns
    -------
    segments

    """
    segments = check_segments(segments, copy=False)
    segments = segment_remove_overlap(segments)
    n = len(segments)
    seg = np.concatenate(([-np.inf], segments.ravel(), [np.inf])).reshape((n + 1, 2))
    if np.all(seg[0, :] == [-np.inf, -np.inf]):
        seg = np.delete(seg, 0, 0)

    if np.all(seg[-1, :] == [np.inf, np.inf]):
        seg = np.delete(seg, -1, 0)

    return seg


def segment_exclusive(segments, *others):
    """Exclusive operation.

    Extracts parts of segments that do not overlap with any other segment.

    Parameters
    ----------
    segments : segment array
    *others : segment arrays

    Returns
    -------
    segments

    """
    # nothing to do if no other segments are provided
    segments = check_segments(segments, copy=False)
    if len(others) == 0:
        return segments

    # combine all other segment lists and invert
    others = segment_union(*others)
    others = segment_invert(others)

    return segment_intersection(segments, others)


def segment_union(*args):
    """Combine segments (logical OR).

    Parameters
    ----------
    *args : segment arrays

    Returns
    -------
    segments

    """

    data = np.zeros((0, 2))

    for obj in args:
        data = np.concatenate((data, check_segments(obj, copy=False)))

    data = segment_remove_overlap(data)

    return data


def segment_difference(*args):
    """Difference between segments (logical XOR).

    Parameters
    ----------
    *args : segment arrays

    Returns
    -------
    segments

    """
    tmp = segment_invert(segment_intersection(*args))
    return segment_intersection(segment_union(*args), tmp)


def segment_intersection(*args):
    """Intersection between segments (logical AND).

    Parameters
    ----------
    *args : segment arrays

    Returns
    -------
    segments

    """
    if len(args) == 0:
        return np.zeros((0, 2))

    segment_list = [segment_remove_overlap(x) for x in args]

    if len(segment_list) == 1:
        return segment_list[0]

    segment_stack = segment_list[0]

    for iseg in segment_list:
        overlap = np.zeros([0, 2])

        for k in range(segment_stack.shape[0]):
            b = np.logical_and(
                segment_stack[k, 0] <= iseg[:, 0], segment_stack[k, 1] > iseg[:, 0]
            )
            b = np.logical_or(
                b,
                np.logical_and(
                    iseg[:, 0] <= segment_stack[k, 0], iseg[:, 1] > segment_stack[k, 0]
                ),
            )
            if sum(b) > 0:
                overlap_start = np.maximum(segment_stack[k, 0], iseg[b, 0])
                overlap_stop = np.minimum(segment_stack[k, 1], iseg[b, 1])
                overlap_new = np.vstack([overlap_start, overlap_stop]).T
                overlap = np.concatenate([overlap, overlap_new])

        if overlap.shape[0] == 0:
            break

        segment_stack = overlap

    return overlap


def segment_scale(segments, value, reference=0.5):
    """Scale segment durations.

    Parameters
    ----------
    segments : segment array
    value : scalar or 1d array
        Scaling factor
    reference: scalar or 1d array
        Relative reference point in segment used for scaling. A value of
        0.5 means symmetrical scaling around the segment center. A value
        of 0. means that the segment duration will be scaled without
        altering the start time.

    Returns
    -------
    segments

    """
    value = np.array(value, dtype=np.float64).squeeze()
    reference = np.array(reference, dtype=np.float64).squeeze()
    segments = check_segments(segments, copy=False)

    if value.ndim == 0:
        value = value.reshape([1])

    if value.ndim == 1:
        value = value.reshape([len(value), 1])
        value = np.diff(segments, axis=1) * (value - 1)
        value = np.concatenate([-reference * value, (1 - reference) * value], axis=1)
        segments = segments + value
    else:
        raise ValueError("Invalid shape of scaling value")

    return segments


def segment_concatenate(*args):
    """Concatenate segments.

    Parameters
    ----------
    *args : segment arrays

    Returns
    -------
    segments

    """
    if len(args) == 0:
        return np.zeros((0, 2))

    segments = np.concatenate([check_segments(x, copy=True) for x in args], axis=0)
    return segments


def segment_count(segments, x):
    """Count number of segments.

    Parameters
    ----------
    segments : segment array
    x : ndarray

    Returns
    -------
    ndarray
        For each value in `x` the number of segments that contain that value.

    """

    segments = check_segments(segments, copy=False)
    x = np.array(x)
    x_shape = x.shape
    x = x.ravel()
    n = segments.shape[0]
    nx = len(x)

    tmp = np.concatenate(
        [np.vstack([segments[:, 0], np.zeros(n)]), np.vstack([x, np.ones(nx)])], axis=1
    )
    tmp = tmp[:, tmp[0, :].argsort(kind="mergesort")]
    idx = tmp[1, :].nonzero()
    tmp_cs = np.cumsum(tmp[1, :], axis=0)
    seg_start = idx - tmp_cs[idx]

    tmp = np.concatenate(
        [np.vstack([segments[:, 1], np.zeros(n)]), np.vstack([x, np.ones(nx)])], axis=1
    )
    tmp = tmp[:, tmp[0, :].argsort(kind="mergesort")]
    idx = tmp[1, :].nonzero()
    tmp_cs = np.cumsum(tmp[1, :], axis=0)
    seg_end = idx - tmp_cs[idx]

    return (seg_start - seg_end).reshape(x_shape)


def segment_overlap(segments, other=None):
    """Returns absolute and relative overlaps between segments.

    Parameters
    ----------
    segments : segment array
    other : segment array, optional
        If `other` is not provided, then overlaps within `segments` are
        analyzed.

    Returns
    -------
    ndarray
        absolute overlap between all combinations of segments
    ndarray
        overlap relative to duration of first segment
    ndarray
        overlap relative to duration of second segment

    """

    segments = check_segments(segments, copy=False)

    other = segments if other is None else check_segments(other, copy=False)

    n_a = len(segments)
    n_b = len(other)

    la = np.diff(segments, axis=1).reshape([n_a, 1])
    lb = np.diff(other, axis=1).reshape([1, n_b])

    delta = np.mean(other, axis=1).reshape([1, n_b]) - np.mean(
        segments, axis=1
    ).reshape([n_a, 1])

    out1 = np.maximum(
        0, np.minimum(-np.abs(delta) + 0.5 * np.abs(lb - la), 0) + np.minimum(la, lb)
    )

    out2 = out1 / la
    out3 = out1 / lb

    return out1, out2, out3


def segment_asindex(segments, x):
    """Convert segments to indices into vector.

    Parameters
    ----------
    segments : segment array
    x : ndarray

    Returns
    -------
    segments (indices)

    """
    x = np.array(x).squeeze()
    b = segment_contains(segments, x)[0]

    b = np.diff(np.concatenate(([0], b, [0])))
    seg = np.vstack(((b == 1).nonzero()[0], (b == -1).nonzero()[0] - 1)).T

    return seg


def segment_join(segments, gap=0):
    """Join segments with small inter-segment gap.

    Parameters
    ----------
    segments : segment array
    gap : scalar
        Segments with an interval equal to or smaller than `gap` will be
        merged.

    Returns
    -------
    segments

    """
    segments = segment_remove_overlap(segments)
    intervals = segments[1:, 0] - segments[:-1, 1]
    idx = (intervals <= gap).nonzero()[0]
    if len(idx) > 0:
        combiseg = np.concatenate((segments[idx, :1], segments[idx + 1, 1:]), axis=1)
        segments = segment_union(segments, combiseg)

    return segments


def segment_split(segments, size=1, overlap=0, join=True, tol=1e-7):
    """Split segments into smaller segments with optional overlap.

    Parameters
    ----------
    segments : segment array
    size : scalar
        Duration of split segments.
    overlap : scalar
        Relative overlap (>=0. and <1.) between split segments.
    join : bool
        Join all split segments into a single segment array. If `join` is
        False, a list is returned with split segments for each original
        segment separately.
    tol : scalar
        Tolerance for determining number of bins.

    Returns
    -------
    segments or list of segments

    """
    segments = check_segments(segments, copy=False)
    nbins = (np.diff(segments, 1, axis=1).ravel() - overlap * size) / (
        (1 - overlap) * size
    )
    idx = (np.ceil(nbins) - nbins) < tol

    nbins[idx] = np.ceil(nbins[idx])
    nbins[~idx] = np.floor(nbins[~idx])
    nbins = nbins.astype(int)

    n = len(nbins)

    seg = []
    if overlap == 0:
        for k in range(0, n):
            tmp = np.arange(0, nbins[k] + 1).reshape((nbins[k] + 1, 1)) * size
            seg.append(segments[k, 0] + np.concatenate((tmp[0:-1], tmp[1:]), axis=1))
    else:
        for k in range(0, n):
            tmp = (
                np.arange(0, nbins[k]).reshape((nbins[k], 1)) * (1 - overlap) * size
                + segments[k, 0]
            )
            seg.append(np.concatenate((tmp, tmp + size), axis=1))

    if join:
        seg = np.zeros((0, 2)) if len(seg) == 0 else np.concatenate(seg, axis=0)

    return seg


def segment_applyfcn(segments, x, *args, **kwargs):
    """Apply function to segmented data.

    Parameters
    ----------
    segments : segment array
    x : ndarray
        The function is applied to values in this array that lie within
        the segments.
    separate : bool
        Apply function to data in each segment separately
    function : callable
        Function that takes one or more data arrays.
    default : any
        Default value for segments that do not contain data (only used
        when separate is True)
    *args : ndarray-like
        Data arrays that are segmented (along first dimension) according
        to the corresponding values in `x` that lie within the segments,
        and passed to `function`.

    Returns
    -------
    ndarray or [ ndarray, ]
        Result of applying function to segmented data.

    """

    b, nn, b2 = segment_contains(segments, x)

    separate = bool(kwargs.get("separate", False))
    function = kwargs.get("function", len)
    default = kwargs.get("default")

    if len(args) == 0:
        if not separate:
            data = function(x[b])
        else:
            data = [
                function(x[ii[0] : (ii[1] + 1)])
                if ii[0] >= 0 and ii[0] <= ii[1]
                else default
                for ii in b2
            ]
    else:
        if not separate:
            data = function(*[y[b] for y in args])
        else:
            data = [
                function(*[y[ii[0] : (ii[1] + 1)] for y in args])
                if ii[0] >= 0 and ii[0] <= ii[1]
                else default
                for ii in b2
            ]

    return data


def segment_uniform_random(segments, size=(1,)):
    """Sample values uniformly from segments.

    Parameters
    ----------
    segments : segment array
    size : tuple of ints
        Shape of returned array

    Returns
    -------
    ndarray

    """

    segments = check_segments(segments, copy=False)

    # calculate segment durations and cumulative sum
    d = np.diff(segments, axis=1).squeeze()
    cs = np.concatenate(([0], np.cumsum(d)))

    # concatenate segments
    s = np.sum(d)

    # draw randomly from concatenated segments
    rtemp = np.random.uniform(low=0.0, high=s, size=size)
    r = np.zeros(rtemp.shape)

    # undo concatenation
    for k in range(len(segments)):
        idx = np.logical_and(rtemp >= cs[k], rtemp < cs[k + 1])
        r[idx] = rtemp[idx] + segments[k, 0] - cs[k]

    return r


class KernelBase:
    def __init__(self):
        pass

    def sample(self, dx):
        raise NotImplementedError


class LinearKernelBase(KernelBase):
    """Base class for linear kernel functions.

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}


    Attributes
    ----------
    kerneltype
    support
    ndim
    bandwidth
    correlation
    covariance

    Methods
    -------
    __call__(dx)

    """

    def __init__(
        self,
        bandwidth=None,
        covariance=None,
        correlation=None,
        kerneltype="symmetrical",
        **kwargs,
    ):
        KernelBase.__init__(self, **kwargs)

        if covariance is not None and (
            bandwidth is not None or correlation is not None
        ):
            raise TypeError

        if covariance is not None:
            self.covariance = covariance
        else:
            self._covariance = self._bw2cov(bandwidth, correlation)

        self._support = 1

        self.kerneltype = kerneltype

    def _cov2bw(self, z):
        if z is None:
            return np.array([1]), np.matrix(1)

        z = np.asmatrix(z, dtype=np.float64)

        bw = np.diag(z) ** 0.5

        d = np.linalg.inv(np.diag(bw))
        p = np.asmatrix(d * z * d, dtype=np.float64)

        return bw, p

    def _bw2cov(self, bw, p):
        if bw is None and p is None:
            return np.matrix(1)

        if bw is None:
            if p.shape[0] != p.shape[1] or np.any(np.abs(p.ravel()) >= 1):
                raise TypeError
            bw = np.ones(len(p))

        if p is None:
            bw = np.asarray(bw, dtype=np.float64)
            if bw.ndim == 0:
                bw = bw.reshape((1,))
            if bw.ndim > 1 or np.any(bw <= 0):
                raise TypeError
            p = np.identity(len(bw))

        p = np.asmatrix(p, dtype=np.float64)

        d = np.diag(bw)
        z = d * p * d

        return z

    @property
    def kerneltype(self):
        """Symmetrical or multiplicative kernel."""
        return self._kerneltype

    @kerneltype.setter
    def kerneltype(self, value):
        if value not in ("symmetrical", "multiplicative"):
            raise TypeError
        self._kerneltype = value

    @property
    def support(self):
        """Finite support"""
        return self._support

    @support.setter
    def support(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError
        self._support = value

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        return len(self._covariance)

    @property
    def bandwidth(self):
        """Kernel bandwidth."""
        bw, p = self._cov2bw(self.covariance)
        return bw

    @bandwidth.setter
    def bandwidth(self, value):
        # convert to array
        value = np.array(value, dtype=np.float64)
        # check if 1D vector
        if value.ndim == 0:
            value = value.reshape((1,))
        elif value.ndim > 1:
            raise TypeError
        # check if same size as covariance matrix
        if len(value) == 1:
            value = value * np.ones(len(self._covariance))
        elif len(value) != len(self._covariance):
            raise TypeError
        if np.any(value <= 0):
            raise TypeError

        self._covariance = self._bw2cov(value, self.correlation)

    @property
    def correlation(self):
        """Correlation matrix."""
        bw, p = self._cov2bw(self.covariance)
        return p

    @correlation.setter
    def correlation(self, value):
        # convert to matrix
        value = np.asmatrix(value, dtype=np.float64)
        # check if square
        if value.shape[0] != value.shape[1]:
            raise TypeError
        # check if symmetric
        if not np.all(value == value.T):
            raise TypeError
        # check if equal size as covariance matrix
        if len(value) != len(self._covariance):
            raise TypeError
        # check if in range <-1,1>
        if np.any(value < -1) or np.any(value > 1) or np.any(np.diag(value) != 1):
            raise TypeError

        self._covariance = self._bw2cov(self.bandwidth, value)

    @property
    def covariance(self):
        """Covariance matrix."""
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        # convert to matrix
        value = np.asmatrix(value, dtype=np.float64)
        # check if square matrix
        if value.shape[0] != value.shape[1]:
            raise TypeError
        # check if symmetric
        if not np.all(value == value.T):
            raise TypeError
        # check if positive semi definite
        if not np.all(np.linalg.eigvalsh(value) > -1e-8):
            raise TypeError

        self._covariance = value

    def _kernel_function(self, u):
        raise NotImplementedError

    def _multiplicative_kernel(self, dx):
        # for each dimension
        # define distance vector
        # compute the 1D kernel
        # and multiply with 1D kernels in other dimensions
        bw, p = self._cov2bw(self._covariance)
        k = np.array(1)
        for idx, b in enumerate(bw):
            u, npoints = self._compute_distance_array(
                np.array([b**2]), np.array([dx[idx]])
            )
            p = self._kernel_function(u)
            k = k[..., np.newaxis] * p.reshape(
                [
                    1,
                ]
                * idx
                + [len(p)]
            )

        return k

    def _symmetrical_kernel(self, dx):
        # define distance matrix
        # compute 1/det(H) * K( H^-1 * x )
        u, npoints = self._compute_distance_array(self._covariance, dx)
        p = self._kernel_function(u)
        p = p.reshape(tuple(npoints))
        return p

    def _compute_distance_array(self, covariance, dx):

        bw, p = self._cov2bw(covariance)
        npoints = np.ceil((self._support * bw) / dx)
        x = [np.arange(-b, b + 1) * a for a, b in zip(dx, npoints, strict=False)]

        if len(x) == 1:
            grid = x[0]
            u = (grid**2) / np.asarray(covariance)
        else:
            grid = np.meshgrid(*x, indexing="ij")
            grid = [g.ravel() for g in grid]
            grid = np.vstack(grid)
            u = np.sum(np.asarray(np.linalg.inv(covariance) * grid) * grid, axis=0)

        return u, npoints.astype(int) * 2 + 1

    def __call__(self, dx):
        """
        Evaluate kernel function.

        Parameters
        ----------
        dx : number or 1D array

        Returns
        -------
        values : ndarray

        """
        dx = np.array(dx, dtype=np.float64)
        if dx.ndim == 0:
            dx = dx.reshape((1,))
        elif dx.ndim > 1:
            raise TypeError

        if len(dx) == 1:
            dx = dx * np.ones(len(self._covariance))
        elif len(dx) != len(self._covariance):
            raise TypeError

        if np.any(dx <= 0):
            raise TypeError

        if self.kerneltype == "multiplicative":
            return self._multiplicative_kernel(dx)
        elif self.kerneltype == "symmetrical":
            return self._symmetrical_kernel(dx)


class GaussianKernel(LinearKernelBase):
    """
    GaussianKernel(
        support=4,
        bandwidth=None,
        covariance=None,
        correlation=None,
        kerneltype="symmetrical",
    )

    Gaussian kernel function.

    Parameters
    ----------
    support : scalar, optional
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    Attributes
    ----------
    kerneltype
    support
    ndim
    bandwidth
    correlation
    covariance

    Methods
    -------
    __call__(dx)

    """

    def __init__(self, support=4, **kwargs):
        LinearKernelBase.__init__(self, **kwargs)
        self.support = support

    def _kernel_function(self, u):
        return np.exp(-0.5 * (u)) / np.sqrt(2 * np.pi)


class EpanechnikovKernel(LinearKernelBase):
    """
    EpanechnikovKernel(
        bandwidth=None, covariance=None, correlation=None, kerneltype="symmetrical"
    )

    Epanechnikov kernel function.

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    Attributes
    ----------
    kerneltype
    ndim
    bandwidth
    correlation
    covariance

    Methods
    -------
    __call__(dx)

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        # u = u**2
        val[u < 1] = 0.75 * (1 - u[u < 1])
        return val


class UniformKernel(LinearKernelBase):
    """
    UniformKernel(
        bandwidth=None, covariance=None, correlation=None, kerneltype="symmetrical"
    )

    Uniform kernel function.

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    Attributes
    ----------
    kerneltype
    ndim
    bandwidth
    correlation
    covariance

    Methods
    -------
    __call__(dx)

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        val[u < 1] = 1.0
        return val


class TriangularKernel(LinearKernelBase):
    """
    TriangularKernel(
        bandwidth=None, covariance=None, correlation=None, kerneltype="symmetrical"
    )

    Triangular kernel function.

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    Attributes
    ----------
    kerneltype
    ndim
    bandwidth
    correlation
    covariance

    Methods
    -------
    __call__(dx)

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        u = np.sqrt(u)
        val[u < 1] = 1 - u[u < 1]
        return val


class VonMisesKernel(KernelBase):
    pass


class MixedKernel(KernelBase):
    """Mixed kernel function.

    Parameters
    ----------
    *args : kernel objects

    Attributes
    ----------
    ndim

    Methods
    -------
    __call__(dx)

    """

    def __init__(self, *args):
        if not np.all([(x is None) or isinstance(x, KernelBase) for x in args]):
            raise TypeError

        self._kernels = list(args)

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        d = 0
        for k in self._kernels:
            d = d + 1 if k is None else d + k.ndim
        return d

    def __call__(self, dx):
        dx = np.array(dx, dtype=np.float64)
        if dx.ndim == 0:
            dx = dx.reshape((1,))
        elif dx.ndim > 1:
            raise TypeError

        if len(dx) == 1:
            dx = dx * np.ones(self.ndim)
        elif len(dx) != self.ndim:
            raise TypeError

        if np.any(dx <= 0):
            raise TypeError

        k = np.array(1)

        for idx, x in enumerate(self._kernels):
            k = k[..., None]

            if x is None:
                pass
            else:
                val = x(dx[idx])
                k = k * val.reshape(
                    [
                        1,
                    ]
                    * (k.ndim - 1)
                    + list(val.shape)
                )

        return k


class NoKernel(KernelBase):
    """Non-smoothing dummy kernel.

    Parameters
    ----------
    ndim : integer

    Attributes
    ----------
    ndim

    """

    def __init__(self, ndim=1):
        self.ndim = ndim

    def __call__(self, dx=1):
        return np.array([1]).reshape(tuple([1] * self._ndim))

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = int(value)


class Smoother:
    """Smoothing class.

    Parameters
    ----------
    kernel : kernel object
    unbiased : bool
        Only take into account available data at edges
    nansaszero : bool
        Treat NaN in data as zeros
    normalize : bool or {'sum','max','none'}
        Method of kernel normalization.

    Attributes
    ----------
    kernel
    unbiased
    nansaszero
    normalization

    Methods
    -------
    __call__(data, dx)

    """

    def __init__(self, kernel=None, unbiased=False, nansaszero=False, normalize=True):
        self.kernel = kernel
        self.unbiased = unbiased
        self.nansaszero = nansaszero
        self.normalize = normalize

    @property
    def kernel(self):
        """Smoothing kernel."""
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        if not isinstance(value, KernelBase):
            raise TypeError
        self._kernel = value

    @property
    def unbiased(self):
        """Whether smoothing is unbiased."""
        return self._unbiased

    @unbiased.setter
    def unbiased(self, value):
        self._unbiased = bool(value)

    @property
    def nansaszero(self):
        """Whether NaNs are reated as zeros."""
        return self._nansaszero

    @nansaszero.setter
    def nansaszero(self, value):
        self._nansaszero = bool(value)

    @property
    def normalize(self):
        """Kernel normalization method."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        if isinstance(value, str):
            if value not in ("sum", "max", "none"):
                raise ValueError("Only 'sum', 'max' and 'none' are supported.")
            else:
                self._normalize = value
        elif value is None:
            self._normalize = "none"
        elif bool(value):
            self._normalize = "sum"
        elif not bool(value):
            self._normalize = "none"
        else:
            raise ValueError("Invalid value.")

    def __call__(self, data, delta=1):
        """Smooth data.

        Parameters
        ----------
        data : ndarray
            The dimensionality of the data should match the dimensionality
            of the kernel.
        delta : scalar or sequence
            sampling interval of the data

        Returns
        -------
        ndarray

        """
        k = self._kernel(delta).copy()

        if self._normalize == "sum":
            k = k / np.nansum(k)
        elif self._normalize == "max":
            k = k / np.nanmax(k)

        if self._nansaszero:
            nan_data = np.isnan(data)
            data = data.copy()
            data[nan_data] = 0

            nan_kernel = np.isnan(k)
            k[nan_kernel] = 0
        else:
            nan_data = None

        data = scipy.signal.convolve(data, k, "same")

        if self._unbiased:
            n = np.ones(data.shape) / np.nansum(k)
            if nan_data is not None:
                n[nan_data] = 0
            n = scipy.signal.convolve(n, k, "same")
            if nan_data is not None:
                n[nan_data] = np.nan
            data = data / n

        return data


_kernel_map = {
    "none": NoKernel,
    "gaussian": GaussianKernel,
    "epanechnikov": EpanechnikovKernel,
    "uniform": UniformKernel,
    "triangular": TriangularKernel,
    "vonmises": VonMisesKernel,
}


def smooth1d(data, axis=-1, kernel="gaussian", bandwidth=1.0, delta=1.0, **kwargs):
    """Smooth array of 1D signals.

    Parameters
    ----------
    data : array
    axis : scalar, optional
        axis of array along which to perform smoothing
    kernel : {'gaussian', 'epanechnikov', 'uniform', 'triangular'}, optional
    bandwidth : scalar, optional
        bandwidth of kernel
    delta : scalar, optional
        sample period of data
    unbiased, nansaszero, normalized : see `Smoother`

    Returns
    -------
    signal : array
        smoothed data array

    """

    data = np.asarray(data)

    k = [
        NoKernel(),
    ] * data.ndim
    k[axis] = _kernel_map[kernel.lower()](bandwidth=bandwidth)

    k = MixedKernel(*k)

    smoother = Smoother(kernel=k, **kwargs)

    data = smoother(data, delta=delta)

    return data


def smooth2d(data, axes=None, kernel="gaussian", bandwidth=1.0, delta=1.0, **kwargs):
    """Smooth array of 2D arrays.

    Parameters
    ----------
    data : array
    axes : 2-element sequence
        the two axes of the data array along which to perform smoothing
    kernel : str or 2-element sequence, optional
        kernel type (one of 'gaussian', 'epanechnikov', 'uniform',
        'triangular') for each of the two dimensions
    bandwidth : scalar or 2-element sequence, optional
        bandwidths of kernel
    delta : scalar or 2-element sequence optional
        sample period of data along the two dimensions
    unbiased, nansaszero, normalized : see `Smoother`

    Returns
    -------
    signal : array
        smoothed data array

    """

    if axes is None:
        axes = [-1, -2]
    data = np.asarray(data)

    bandwidth = np.array(bandwidth).ravel()

    if data.ndim < 2:
        raise ValueError("Require at least 2 dimensions.")

    if len(axes) != 2 or axes[0] == axes[-1]:
        raise ValueError("Invalid axes")

    if not isinstance(kernel, (list, tuple)):
        kernel = [
            kernel,
        ]

    if len(kernel) < 1 or len(kernel) > 2:
        raise ValueError("Specify at least and at most 2 kernels for the 2 dimensions.")

    for k in kernel:
        if k not in list(_kernel_map.keys()):
            raise ValueError("Unknown kernel")

    k = [
        NoKernel(),
    ] * data.ndim
    k[axes[0]] = _kernel_map[kernel[0].lower()](bandwidth=bandwidth[0])
    k[axes[1]] = _kernel_map[kernel[-1].lower()](bandwidth=bandwidth[-1])

    k = MixedKernel(*k)

    smoother = Smoother(kernel=k, **kwargs)

    data = smoother(data, delta=delta)

    return data


class SegmentError(Exception):
    """Exception raised if array does not represent segments"""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


def partition_vector(v, **kwargs):
    """Partitions vector into subvectors.

    See :func:`partitions` for detailed help on keyword arguments.

    Parameters
    ----------
    v : array_like, can be indexed with numpy array

    Returns
    -------
    tuple of subvectors

    See Also
    --------
    partitions

    """
    kwargs["size"] = len(v)
    p = partitions(**kwargs)
    return (v[idx] for idx in p)


def partitions(
    size=None, partsize=None, nparts=None, method="block", keepremainder=True
):
    """Partition elements in multiple groups.

    Parameters
    ----------
    size : int, optional
        Number of elements to partition. If not given, `size` will be
        calculated as `nparts` * `partsize`
    partsize : int, optional
        Number of elements in a partition. If not given, 'partsize' will
        be calculated as ceil( `size` / `nparts` ).
    nparts : int, optional
        Number of partitions. If not given, 'nparts' will be calculated
        as ceil( `size` / `partsize` ).
    method : {'block','random','sequence'}, optional
        Partitioning method. 'block': first `partsize` elements are
        assigned to partition 1, second `partsize` elements to partition
        2, etc. 'random': each elements is randomly assigned to a
        partition. 'sequence': elements are distributed to partitions in
        order.
    keepremainder : bool, optional
        Whether or not to keep remaining elements that are not part of a
        partition (default is True).

    Returns
    -------
    tuple of 1D arrays
        Each array contains the indices of the elements in a partition

    """
    args = [size is None, partsize is None, nparts is None]
    if all(args) or not any(args):
        raise ValueError

    if size is None:
        if partsize is None:
            partsize = 1
        elif nparts is None:
            nparts = 1
        size = nparts * partsize
    elif partsize is None:
        if nparts is None:
            partsize = 1
            nparts = size
        else:
            partsize = np.int(np.ceil(size / nparts))
    else:
        nparts = np.int(np.ceil(size / partsize))

    if nparts * partsize > size and not keepremainder:
        if args[1]:  # partsize was None
            partsize = np.int(np.floor(size / nparts))
        elif args[2]:  # nparts was None
            nparts = np.int(np.floor(size / partsize))
        else:
            raise ValueError(
                "Cannot keep remainder, but size is not divisible by partsize or nparts"
            )
        size = nparts * partsize

    if method == "block":
        idx = np.floor(np.arange(size) / partsize)
    elif method == "random":
        idx = np.floor(np.arange(size) / partsize)
        np.random.shuffle(idx)
    elif method == "sequence":
        idx = np.remainder(np.arange(size), nparts)
    else:
        raise TypeError("Method argument should be one of block, random, sequence")

    return (np.nonzero(idx == k)[0] for k in np.arange(nparts))


def _issorted(x, strict):
    n = x.shape[0]
    flag = True

    if n < 3:
        flag = True
    else:
        if strict:
            if x[1] > x[0]:
                for k in range(1, n):
                    if x[k + 1] <= x[k]:
                        flag = False
                        break
            elif x[1] < x[0]:
                for k in range(1, n):
                    if x[k + 1] >= x[k]:
                        flag = False
                        break
            else:
                flag = False
        else:
            for k in range(0, n - 1):
                if x[k] != x[k + 1]:
                    pos = x[k] < x[k + 1]
                    for j in range(k + 1, n - 1):
                        if (x[j] > x[j + 1] and pos) or (x[j] < x[j + 1] and not pos):
                            flag = False
                            break
                    break

    return flag


def issorted(x, strict=False):
    """
    Tests if vector is sorted.

    Parameters
    ----------
    x : array_like
        will be converted into a vector before testing if it is sorted.
    strict : bool, optional
        values in `x` should be strictly monotonically increasing or
        decreasing (i.e. vectors with equal values are not considered
        strictly sorted).

    Returns
    -------
    bool
        whether input vector is sorted.

    See Also
    --------
    isascending, isdescending

    """
    x = np.asarray(x).ravel()
    return _issorted(x, strict)


class Segment:
    """Segment container class.

    Parameters
    ----------
    data : Segment, array_like

    """

    def __init__(self, data=None, copy=True):
        if data is None:
            data = []
        if isinstance(data, Segment):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else:
            self._data = check_segments(data, copy)

    @classmethod
    def issegment(cls, x):
        """Test is `x` is valid segment array."""

        if isinstance(x, Segment):
            return True

        try:
            check_segments(x)
        except ValueError:
            return False

        return True

    @classmethod
    def fromarray(cls, data):
        """Construct Segment from array.

        Parameters
        ----------
        data : (n,2) array

        Returns
        -------
        Segment

        """
        return Segment(data)

    @classmethod
    def fromlogical(cls, y, x=None):
        """Construct Segment from logical vector.

        Parameters
        ----------
        y : 1d logical array
            Any sequenceof True values that is flanked by False values is
            converted into a segment.
        x : 1d array like, optional
            The segment indices from `y` will be used to index into `x`.

        Returns
        -------
        Segment

        """

        y = np.asarray(y, dtype=np.int8)

        if len(y) == 0 or np.all(y == 0):
            return Segment([])

        d = np.diff(np.concatenate(([0], y, [0])))
        segstart = np.nonzero(d[0:-1] == 1)[0]
        segend = np.nonzero(d[1:] == -1)[0]

        seg = np.vstack((segstart, segend)).T

        if x is not None:
            seg = x[seg]

        return Segment(seg)

    @classmethod
    def fromindices(cls, y, x=None):
        """Construct segments from vector of indices.

        Parameters
        ----------
        y : 1d array like
            Vector of indices. Segments are created from all neighboring
            pairs of values in y (as long as the difference is positive).
        x : 1d array like, optional
            The segment indices from `y` will be used to index into `x`.

        Returns
        -------
        Segment

        """

        if len(y) == 0:
            return Segment([])

        d = np.nonzero(np.diff(y) > 1)[0]
        segstart = y[np.concatenate(([0], d + 1))][: len(y) - 1]
        segend = y[np.concatenate((d + 1, [len(y) - 1]))][: len(y) - 1]

        seg = np.vstack((segstart, segend)).T

        if x is not None:
            seg = x[seg]

        return Segment(seg)

    @classmethod
    def fromevents(cls, on, off, greedy_start=False, greedy_stop=False):
        """Construct segments from sequences of start and stop values.

        Parameters
        ----------
        on : 1d array like
            segment start values.
        off : 1d array like
            segment stop values.
        greedyStart : bool
            If multiple start values precede a stop value, then the first
            start value is used.
        greedyStop : bool
            If multiple stop values follow a start value, then the last
            stop value is used.

        Returns
        -------
        Segment

        """

        on = np.array(on, dtype=np.float64).ravel()
        off = np.array(off, dtype=np.float64).ravel()

        events = np.concatenate((on, off))
        eventid = np.concatenate((np.ones(len(on)), -np.ones(len(off))))

        isort = np.argsort(
            events, kind="mergesort"
        )  # mergesort keeps items with same key in same relative order
        events = events[isort]
        eventid = eventid[isort]

        diff_eventid = np.diff(eventid)

        # if greedyStart = True, remove all on-events in blocks (except first one)
        if greedy_start:
            invalid = (
                np.nonzero(np.logical_and(diff_eventid == 0, eventid[1:] == 1))[0] + 1
            )
        else:
            invalid = np.nonzero(np.logical_and(diff_eventid == 0, eventid[0:-1] == 1))[
                0
            ]

        # if greedyStop = True, remove all off-events in blocks (except last one)
        if greedy_stop:
            invalid = np.concatenate(
                (
                    invalid,
                    np.nonzero(np.logical_and(diff_eventid == 0, eventid[0:-1] == -1))[
                        0
                    ],
                )
            )
        else:
            invalid = np.concatenate(
                (
                    invalid,
                    np.nonzero(np.logical_and(diff_eventid == 0, eventid[1:] == -1))[0]
                    + 1,
                )
            )

        events = np.delete(events, invalid)
        eventid = np.delete(eventid, invalid)

        s = np.nonzero(np.diff(eventid) == -2)[0]
        s = np.vstack((events[s], events[s + 1])).T

        return Segment(s)

    @classmethod
    def fromduration(cls, anchor, duration, reference=0.5):
        """Construct segments from anchor points and durations.

        Parameters
        ----------
        anchor : scalar or 1d array like
            Anchoring points for the new segments. If `reference` is not
            given, then the anchor determines the segment center.
        duration : scalar or 1d array like
            Durations of the new segments
        reference : scalar or 1d array like, optional
            Relative reference point of the anchor in the segment.
            If `reference` is 0., the anchor defines the segment start,
            if `reference` is 1., the anchor defines the segment stop.

        Returns
        -------
        Segment

        """

        # anchor + duration*[-reference (1-reference)]
        anchor = np.array(anchor, dtype=np.float64).ravel()
        duration = np.array(duration, dtype=np.float64).ravel()
        reference = np.array(reference, dtype=np.float64).ravel()

        start = anchor - reference * duration
        stop = anchor + (1 - reference) * duration

        return Segment(np.vstack((start, stop)).T)

        pass

    def __array__(self, *args):
        return self._data.__array__(*args)

    def asarray(self):
        """Return numpy array representation of Segment object data."""
        return self._data  # should we return a copy here?

    def __repr__(self):
        """Return string representation of Segment object."""
        return "Segment(" + repr(self._data) + ")"

    def __str__(self):
        """Return string representation of Segment object data."""
        return "Segment(" + str(self._data) + ")"

    @property
    def start(self):
        """Return a vector of segment start values."""
        # this returns a copy
        return self._data[:, 0].copy()

    @start.setter
    def start(self, value):
        """Set segment start values."""
        # Should we re-order after changing start points?
        if np.any(self._data[:, 1] < value):
            raise SegmentError("Segment start times should be <= stop times")

        self._data[:, 0] = value

    @property
    def stop(self):
        """Return a vector of segment stop values."""
        # this returns a copy
        return self._data[:, 1].copy()

    @stop.setter
    def stop(self, value):
        """Set segment stop values."""
        # TODO: check if values are not beyond start points
        if np.any(self._data[:, 0] > value):
            raise SegmentError("Segment stop times should be >= start times")

        self._data[:, 1] = value

    @property
    def duration(self):
        """Return a vector of segment durations."""
        return np.diff(self._data, axis=1).ravel()

    @duration.setter
    def duration(self, value):
        """Set new duration of segments."""
        value = np.array(value, dtype=np.float64).ravel()
        ctr = np.mean(self._data, axis=1)
        self._data[:, 0] = ctr - 0.5 * value
        self._data[:, 1] = ctr + 0.5 * value

    @property
    def center(self):
        """Return a vector of segment centers."""
        return np.mean(self._data, axis=1)

    @center.setter
    def center(self, value):
        """Set new centers of segments."""

        value = np.array(value, dtype=np.float64).ravel()
        dur = np.diff(self._data, axis=1).squeeze()
        self._data[:, 0] = value - 0.5 * dur
        self._data[:, 1] = value + 0.5 * dur

    def __len__(self):
        """Return the number segments in the container."""
        return int(self._data.shape[0])

    def issorted(self):
        """Check if segment starts are sorted in ascending order."""
        return issorted(self._data[:, 0])

    def isort(self):
        """Sort segments (in place) in ascending order according to start value."""
        if self._data.shape[0] > 1:
            idx = np.argsort(self._data[:, 0])
            self._data = self._data[idx, :]

        return self

    def sort(self):
        """Sort segments in ascending order according to start value.

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.isort()
        return s

    def argsort(self):
        """Argument sort of segment start value.

        Returns
        -------
        ndarray
            Indices that will sort the segment array.

        """
        return np.argsort(self._data[:, 0])

    @property
    def intervals(self):
        """Duration of intervals between segments."""
        return self._data[1:, 0] - self._data[:-1, 1]

    def hasoverlap(self):
        """Check if any segments are overlapping."""
        return segment_has_overlap(self._data)

    def removeoverlap(self, strict=True):
        """Remove overlap between segments through merging.

        This method will sort segments as a side effect.

        Parameters
        ----------
        strict : bool
            Only merge two segments if the end time of the first is stricly
            larger than (and not equal to) the start time of the second segment.

        Returns
        -------
        Segment

        """
        self._data = segment_remove_overlap(self._data, strict=strict)
        return self

    def __iter__(self):
        """Iterate through segments in container."""
        idx = 0
        while idx < self._data.shape[0]:
            yield self._data[idx, 0], self._data[idx, 1]
            idx += 1

    def not_(self):
        """Test if no segments are defined."""
        return self._data.shape[0] == 0

    def truth(self):
        """Test if one or more segments are defined."""
        return self._data.shape[0] > 0

    def exclusive(self, *others):
        """Exclude other segments.

        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iexclusive(*others)
        return s

    def iexclusive(self, *others):
        """Exclude other segments (in place).

        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_exclusive(self._data, *others)
        return self

    def invert(self):
        """Invert segments.

        Constructs segments from the inter-segment intervals.
        This method Will remove overlap as a side effect.

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iinvert()
        return s

    def iinvert(self):
        """Invert segments (in place).

        Constructs segments from the inter-segment intervals.
        This method Will remove overlap as a side effect.

        """
        self._data = segment_invert(self._data)
        return self

    __invert__ = invert

    def union(self, *others):
        """Combine segments (logical OR).

        This method Will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iunion(*others)
        return s

    def iunion(self, *others):
        """Combine segments (logical OR) (in place).

        This method Will remove overlaps as a sife effect.

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_union(self._data, *others)
        return self

    __or__ = union
    __ror__ = __or__
    __ior__ = iunion

    def difference(self, *others):
        """Return non-overlapping parts of segments (logical XOR).

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.idifference(*others)
        return s

    def idifference(self, *others):
        """Return non-overlapping parts of segments (logical XOR) (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_difference(self._data, *others)
        return self

    def __xor__(self, other):
        """Return non-overlapping parts of segments (logical XOR).

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        return self.difference(other)

    __rxor__ = __xor__
    __ixor__ = idifference

    def intersection(self, *others):
        """Return intersection (logical AND) of segments.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iintersection(*others)
        return s

    def iintersection(self, *others):
        """Return intersection (logical AND) of segments (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_intersection(self._data, *others)
        return self

    __and__ = intersection
    __rand__ = __and__
    __iand__ = iintersection

    def __eq__(self, other):
        """Test if both objects contain the same segment data.

        Parameters
        ----------
        other : segment array

        Returns
        -------
        bool

        """
        if not isinstance(other, Segment):
            other = Segment(other)
        return (self._data.shape == other._data.shape) and np.all(
            self._data == other._data
        )

    def __ne__(self, other):
        """Test if objects contain dissimilar segment data.

        Parameters
        ----------
        other : segment array

        Returns
        -------
        bool

        """
        if not isinstance(other, Segment):
            other = Segment(other)
        return (self._data.shape != other._data.shape) and np.any(
            self._data != other._data
        )

    def __getitem__(self, key):
        """Slice segments.

        Parameters
        ----------
        key : slice or indices

        Returns
        -------
        Segment

        """
        return Segment(self._data[key, :])  # does not return a view!

    def __setitem__(self, key, value):
        """Set segment values.

        Parameters
        ----------
        key : slice or indices
        value : scalar or ndarray

        """
        self._data[key, :] = value
        return self

    def __delitem__(self, key):
        """Delete segments (in place).

        Parameters
        ----------
        key : array like
            Index vector or boolean vector that indicates which segments
            to delete.

        """
        # make sure we have a np.ndarray
        key = np.array(key)

        # if a logical vector with length equal number of segments, then find indices
        if key.dtype == np.bool and key.ndim == 1 and len(key) == self._data.shape[0]:
            key = np.nonzero(key)[0]

        self._data = np.delete(self._data, key, axis=0)
        return self

    def offset(self, value):
        """Add offset to segments.

        Parameters
        ----------
        value : scalar or 1d array

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.ioffset(value)

        return s

    def ioffset(self, value):
        """Add offset to segments (in place).

        Parameters
        ----------
        value : scalar or 1d array

        """
        value = np.array(value, dtype=np.float64).squeeze()

        if value.ndim == 1:
            value = value.reshape([len(value), 1])
        elif value.ndim != 0:
            raise ValueError("Invalid shape of offset value")

        self._data = self._data + value
        return self

    def scale(self, *args, **kwargs):
        """Scale segment durations.

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iscale(*args, **kwargs)
        return s

    def iscale(self, value, reference=0.5):
        """Scale segment durations (in place).

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.

        """
        self._data = segment_scale(self._data, value, reference=reference)
        return self

    def concat(self, *others):
        """Concatenate segments.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.iconcat(*others)
        return s

    def iconcat(self, *others):
        """Concatenate segments (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_concatenate(self._data, *others)
        return self

    def __iadd__(self, value):
        """Concatenates segments or adds offset (in place).

        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If `value` is a Segment object, then its segments are
            concatenated to this Segment. Otherwise,
            `value` is added as an offset to the segments.

        """
        if isinstance(value, Segment):
            return self.iconcat(value)

        return self.ioffset(value)

    def __add__(self, value):
        """Concatenate segments or adds offset.

        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If `value` is a Segment object, then a new Segment object
            with a concatenated list of segment is returned. Otherwise,
            `value` is added as an offset to the segments.

        Returns
        -------
        Segment

        """
        if isinstance(value, Segment):
            return self.concat(value)

        return self.offset(value)

    __radd__ = __add__

    def __sub__(self, value):
        """Subtract value.

        Parameters
        ----------
        value : scalar or 1d array

        Returns
        -------
        Segment

        """
        return self.offset(-value)

    __rsub__ = __sub__

    def __isub__(self, value):
        """Subtract value (in place).

        Parameter
        ---------
        value : scalar or 1d array

        """
        return self.ioffset(-value)

    __mul__ = scale
    __rmul__ = __mul__
    __imul__ = iscale

    def __truediv__(self, value):
        """Divide segment durations.

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.

        Returns
        -------
        Segment

        """
        return self.scale(1.0 / value)

    def __rtruediv__(self, value):
        return NotImplemented

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __itruediv__(self, value):
        """Divide segment durations (in place).

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.

        """
        return self.iscale(1.0 / value)

    __idiv__ = __itruediv__

    def contains(self, value, issorted=True, expand=None):
        """Test if values are contained in segments.

        Segments are considered left closed and right open intervals. So,
        a value x is contained in a segment if start<=x and x<stop.

        Parameters
        ----------
        value : sorted 1d array
        issorted : bool
            Assumes vector `x` is sorted and will not sort it internally.
            Note that even if `issorted` is False, the third output argument
            will still return indices into the (internally) sorted vector.
        expand : bool
            Will expand the last output to full index arrays into 'x' for
            each segment. The default is True if `issorted` is False and
            vice versa. Note that for non-sorted data (`issorted` is False) and
            `expand`=False, the last output argument will contain start and stop
            indices into the (internally) sorted input array.

        Returns
        -------
        ndarray
            True for each value in `x` that is contained within any segment.
        ndarray
            For each segment the number of values in `x` that it contains.
        ndarray
            For each segment, the start and end indices of values in `x`
            that are contained within that segment.

        """
        # TODO: test if self is sorted
        # TODO: test if value is sorted
        # TODO: support scalars and nd-arrays for value?
        return segment_contains(self._data, value, issorted, expand)

    def __contains__(self, value):
        return self.contains(value)[0]

    def count(self, x):
        """Count number of segments.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
            For each value in `x` the number of segments that contain that value.

        """
        return segment_count(self._data, x)

    def overlap(self, other=None):
        """Returns absolute and relative overlaps between segments.

        Parameters
        ----------
        other : segment array, optional
            If `other` is not provided, then auto-overlaps are analyzed.

        Returns
        -------
        ndarray
            absolute overlap between all combinations of segments
        ndarray
            overlap relative to duration of first segment
        ndarray
            overlap relative to duration of second segment

        """
        return segment_overlap(self._data, other=other)

    def asindex(self, x):
        """Convert segments to indices into vector.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        Segment (indices)

        """
        return Segment(segment_asindex(self._data, x))

    def ijoin(self, gap=0):
        """Join segments with small inter-segment gap (in place).

        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than `gap` will be
            merged.

        """
        self._data = segment_join(self._data, gap=gap)
        return self

    def join(self, *args, **kwargs):
        """Join segments with small inter-segment gap.

        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than `gap` will be
            merged.

        Returns
        -------
        Segment

        """
        s = Segment(self)
        s.ijoin(*args, **kwargs)
        return s

    def split(self, size=1, overlap=0, join=True, tol=1e-7):
        """Split segments into smaller segments with optional overlap.

        Parameters
        ----------
        size : scalar
            Duration of split segments.
        overlap : scalar
            Relative overlap (>=0. and <1.) between split segments.
        join : bool
            Join all split segments into a single segment array. If `join` is
            False, a list is returned with split segments for each original
            segment separately.
        tol : scalar
            Tolerance for determining number of bins.

        Returns
        -------
        Segment or list of Segments

        """

        seg = segment_split(self._data, size=size, overlap=overlap, join=join, tol=tol)
        if len(seg) == 0:
            return Segment([])
        elif isinstance(seg, list):  # we have a list of segments
            return [Segment(x) for x in seg]
        else:
            return Segment(seg)

    def applyfcn(self, x, *args, **kwargs):
        """Apply function to segmented data.

        Parameters
        ----------
        x : ndarray
            The function is applied to values in this array that lie within
            the segments.
        separate : bool
            Apply function to data in each segment separately
        function : callable
            Function that takes one or more data arrays.
        default : any
            Default value for segments that do not contain data (only used
            when separate is True)
        *args : ndarray-like
            Data arrays that are segmented (along first dimension) according
            to the corresponding values in `x` that lie within the segments,
            and passed to `function`.

        Returns
        -------
        ndarray or [ ndarray, ]
            Result of applying function to segmented data.

        """
        return segment_applyfcn(self._data, x, *args, **kwargs)

    def partition(self, **kwargs):
        """Partition segments into groups.

        See `fklab.general.partitions` and `fklab.general.partition_vector'
        for more information.

        Parameters
        ----------
        nparts : int
            Number of partitions
        method: 'block', 'random', 'sequence'
            Method of assigning segments to partitions.

        Returns
        -------
        Segment object
            partitioned subset of segments

        """
        return partition_vector(self, **kwargs)
        # kwargs['size'] = self._data.shape[0]
        # return (self[idx] for idx in partitions( **kwargs ))

    def uniform_random(self, size=(1,)):
        """Sample values uniformly from segments.

        Parameters
        ----------
        size : tuple of ints
            Shape of returned array.

        Returns
        -------
        ndarray

        """
        return segment_uniform_random(self._data, size=size)


def inrange(x, low=None, high=None, include_boundary=True):
    """Tests if values are in range.

    The range is defined by a lower (`low`) and upper (`high`) boundary.
    A value of None indicates no boundary. If the upper boundary is
    smaller than the lower boundary, the the range is inverted. For example,
    if low=10 and high=5, then all values in x that are smaller than 5 or
    larger than 10 are within range.

    Parameters
    ----------
    x : array-like
        data values to test
    low : scalar number
        lower boundary of the range, default is `None` (no lower boundary)
    high : scalar number
        upper boundary of the range, default is `None` (no upper boundary)
    include_boundary : bool
        whether or not the boundaries are included, default is `True`

    Returns
    -------
    bool array
        True for all values in x that are within range

    Examples
    --------
    >>> b = inrange( [1,2,3,4,5] )
    array([ True,  True,  True,  True,  True], dtype=bool)

    >>> b = inrange( [1,2,3,4,5], low=3 )
    array([ False,  False,  True,  True,  True], dtype=bool)

    >>> b = inrange( [1,2,3,4,5], high=2 )
    array([ True,  True, False, False, False], dtype=bool)

    >>> b = inrange( [1,2,3,4,5], low=2, high=4 )
    array([False,  True,  True,  True, False], dtype=bool)

    >>> b = inrange( [1,2,3,4,5], low=4, high=2 )
    array([ True,  True, False,  True,  True], dtype=bool)

    """

    x = np.asarray(x)

    if include_boundary:
        op_low = np.greater_equal
        op_high = np.less_equal
    else:
        op_low = np.greater
        op_high = np.less

    if low is None:
        if high is None:
            return np.ones(x.shape, dtype=bool)
        else:
            return op_high(x, high)

    if high is None:
        return op_low(x, low)

    if high >= low:
        return np.logical_and(op_low(x, low), op_high(x, high))
    else:
        return np.logical_or(op_low(x, low), op_high(x, high))


standard_frequency_bands = {
    "slow": [0.1, 1.0],
    "delta": [1.0, 4.0],
    "theta": [6.0, 12.0],
    "spindle": [7.0, 14.0],
    "beta": [15.0, 30.0],
    "gamma": [30.0, 140.0],
    "gamma_low": [30.0, 50.0],
    "gamma_high": [60.0, 140.0],
    "ripple": [140.0, 225.0],
    "mua": [300.0, 2000.0],
    "hfo": [80.0, 500.0],
}


def _zerocrossing(y):
    # find all positive and negative values
    sy = np.sign(y)
    isy = np.flatnonzero(sy)
    sy = sy[isy]

    # compute difference: negative values indicate positive-to-negative
    # transitions and positive values indicate negative-to-positive
    # transitions
    dsy = np.diff(sy)

    p2n = np.flatnonzero(dsy < 0)
    n2p = np.flatnonzero(dsy > 0)

    # look up the start and end indices that span the transitions
    p2n_x1 = isy[p2n] + 1
    p2n_x2 = isy[p2n + 1] - 1

    n2p_x1 = isy[n2p] + 1
    n2p_x2 = isy[n2p + 1] - 1

    # compute the (fractional) index of the transition
    p2n = (p2n_x1 + p2n_x2) / 2.0
    n2p = (n2p_x1 + n2p_x2) / 2.0

    # todo: correct index
    mask = p2n_x2 < p2n_x1
    a = np.abs(y[p2n_x2[mask]] / y[p2n_x1[mask]])
    p2n[mask] = p2n[mask] + 0.5 * (p2n_x1[mask] - p2n_x2[mask]) * (a - 1) / (a + 1)

    mask = n2p_x2 < n2p_x1
    a = np.abs(y[n2p_x2[mask]] / y[n2p_x1[mask]])
    n2p[mask] = n2p[mask] + 0.5 * (n2p_x1[mask] - n2p_x2[mask]) * (a - 1) / (a + 1)

    return p2n, n2p


def zerocrossing(y, x=None):
    """Detects zero crossings in 1D data array.

    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.

    Returns
    -------
    p2n : array
        index at positive-to-negative zero crossings
    n2p: array
        index at negative-to-positive zero crossings

    """
    p2n, n2p = _zerocrossing(y)

    if x is not None:
        p2n = np.interp(p2n, np.arange(len(y)), x)
        n2p = np.interp(n2p, np.arange(len(y)), x)

    return p2n, n2p


def _localextrema_gradient(y):
    return zerocrossing(np.gradient(y))


def _localextrema_discrete(y):

    dy = np.sign(np.diff(y))
    inz = np.flatnonzero(dy)
    dy = dy[inz]
    ddy = np.diff(dy)
    imax = np.flatnonzero(ddy < 0)
    imin = np.flatnonzero(ddy > 0)

    imax = (1 + inz[imax] + inz[imax + 1]) / 2.0
    imin = (1 + inz[imin] + inz[imin + 1]) / 2.0

    return imax, imin


def localextrema(
    y, x=None, method="discrete", kind="extrema", yrange=None, interp="linear"
):
    """Detects local extrema (maxima and/or minima) in 1D data array.

    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.
    method : str
        method for computing extrema, one of ``discrete`` or ``gradient``.
        Default is ``discrete``.
    kind : str
        type of extrema to compute, one of ``extrema``, ``extremes``,
        ``max``, ``maximum``, ``maxima``, ``min``, ``minimum``, ``minima``
        Default is ``extrema``
    yrange : 2-element sequence (optional)
        range of acceptable y-values for the absolute extrema. Default is None.
    interp : callable or str
        if callable, it should define an interpolation function that takes
        a 1D array of values and an array of indices at which to interpolate.
        If a string, it specifies the kind of interpolation for the
        scipy.interpolation.interp1d function. Default is ``linear``.

    Returns
    -------
    index : array
        if x is None, an array of (possibly fractional) indices at which
        the extrema where detected. If x is given, then the interpolated
        values in x are returned.
    value: array
        interpolated y-values at the detected extrema.

    """

    # compute the local extrema
    if method in ["gradient"]:
        imax, imin = _localextrema_gradient(y)
    elif method in ["discrete"]:
        imax, imin = _localextrema_discrete(y)
    else:
        raise LookupError

    # select the requested extrema
    if kind in ("extrema", "extremes"):
        ii = np.sort(np.concatenate((imax, imin)))
    elif kind in ("max", "maximum", "maxima"):
        ii = imax
    elif kind in ("min", "minimum", "minima"):
        ii = imin
    else:
        raise LookupError

    # compute signal amplitude at local extrema
    try:
        amp = interp(y, ii)  # type: ignore
    except TypeError:
        amp = sp.interpolate.interp1d(np.arange(len(y)), y, kind=interp, copy=False)(ii)

    # apply threshold on absolute amplitude
    if yrange is not None:
        valid = inrange(np.abs(amp), low=yrange[0], high=yrange[-1])
        ii = ii[valid]
        amp = amp[valid]

    # interpolate x values
    if x is not None:
        ii = np.interp(ii, np.arange(len(y)), x)

    return ii, amp


def localmaxima(y, **kwargs):
    """Detects local maxima in 1D data array.

    See also
    --------
    localextrema

    """

    kwargs["kind"] = "max"
    return localextrema(y, **kwargs)


def detect_mountains(y, x=None, low=None, high=None, segments=None):
    """Detects segments with above-threshold values 1D data array.

    Parameters
    ----------
    y : array
        array of data values
    x : array
        array of index values. Default is None.
    low : scalar number
        lower data value threshold
    high : scalar number
        upper data value threshold
    segments : Segment (optional)
        pre-defined segments in which to search for above-threshold values

    Returns
    -------
    seg : Segment
        list of segments that meet the threshold conditions

    """

    # define indices, if not given
    if x is None:
        x = np.arange(len(y))

    if low is None:
        if high is None:
            # compute 90% percentile if no thresholds are given
            if segments is None:
                low = np.percentile(y, 90)
            else:
                low = segments.applyfcn(x, y, function=lambda z: np.percentile(z, 90))
        else:
            # only one threshold is given
            low = high
            high = None

    # find all segments in which y is above lower threshold
    s = Segment.fromlogical(y > low, x)

    if len(s) > 0:
        # for each segment, test if maximum y-value is above upper threshold
        if high is not None:
            valid = s.applyfcn(
                x, y, function=lambda z: np.max(z) > high, separate=True, default=False
            )
            del s[~np.array(valid)]

        # combine with user-provided segments
        if segments is not None:
            s = s & segments

    return s


def construct_filter(band, fs=1.0, transition_width="25%", attenuation=60):
    """Constructs FIR high/low/band-pass filter.

    Parameters
    ----------
    band : str, scalar or 2-element sequence
        either a valid key into the `default_frequency_bands` dictionary,
        a scalar for a low-pass filter, or a 2-element sequence with lower
        and upper pass-band frequencies. Use 0., None, Inf or NaN for the
        lower/upper cut-offs in the sequence to define a low/high-pass filter.
        If band[1]<band[0], then a stop-band filter is constructed.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.

    Returns
    -------
    1D array
        filter coefficients

    """

    # look up pre-defined frequency band
    if isinstance(band, str):
        band = standard_frequency_bands[band]

    band = np.array(band, dtype=np.float64).ravel()

    if len(band) == 1:
        # scalar -> low=pass filter
        band = np.array([0.0, float(band)], dtype=np.float64)
    elif len(band) != 2:
        raise ValueError("Invalid frequency band")

    if np.diff(band) == 0.0:
        raise ValueError("Identical frequencies not allowed.")

    lower, upper = np.logical_or.reduce((np.isnan(band), np.isinf(band), band <= 0.0))

    if not lower and upper:
        # high pass filter
        band = band[0]
        pass_zero = False
        band_width = fs / 2.0 - band
    elif not upper and lower:
        # low pass filter
        band = band[1]
        pass_zero = True
        band_width = band
    elif lower and upper:
        raise ValueError("Invalid frequency band")
    else:
        pass_zero = np.diff(band)[0] < 0
        if pass_zero:
            band = band[::-1]
        band_width = np.diff(band)[0]

    if fs <= 2 * np.max(band):
        raise ValueError("Frequency band too high for given sampling frequency")

    if isinstance(transition_width, str):
        transition_width = band_width * float(transition_width.rstrip("%")) / 100.0

    n, beta = sp.signal.kaiserord(attenuation, transition_width * 2.0 / fs)

    # always have odd n
    n = n + (n + 1) % 2

    h = sp.signal.firwin(
        n, band, window=("kaiser", beta), pass_zero=bool(pass_zero), scale=False, fs=fs
    )

    return h


def apply_filter(signal, band, axis=-1, **kwargs):
    """Applies low/high/band-pass filter to signal.

    Parameters
    ----------
    signal : array
    band : str, scalar or 2-element sequence
        frequency band, either as a string, a scalar or [low,high] sequence.
        See `construct_filter` for more details.
    axis : scalar, optional
        axis along which to filter
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB

    Returns
    -------
    array
        filtered signal

    """

    b = construct_filter(band, **kwargs)

    if isinstance(signal, (tuple, list)):
        signal = [sp.signal.filtfilt(b, 1.0, np.asarray(x), axis=axis) for x in signal]
    else:
        signal = np.asarray(signal)
        signal = sp.signal.filtfilt(b, 1.0, signal, axis=axis)

    return signal


def compute_envelope(
    signals,
    freq_band=None,
    axis=-1,
    fs=1.0,
    isfiltered=False,
    filter_options=None,
    smooth_options=None,
    pad=True,
):
    """Computes average envelope of band-pass filtered signal.

    Parameters
    ----------
    signals : array
        either array with raw signals (`isfiltered`==False) or
        pre-filtered signals (`isfiltered`==True). Can also be a sequence
        of such signals.
    freq_band : str or 2-element sequence, optional
        frequency band (in case signal needs to filtered)
    axis : scalar, optional
        axis of the time dimension in the signals array
    fs : scalar, optional
        sampling frequency
    isfiltered : bool, optional
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
        See `apply_filter` and `construct_filter`.
    smooth_options : dict, optional
        dictionary with optional kernel and bandwidth keys for envelope
        smoothing (see `fklab.signals.kernelsmoothing.smooth1d`)
    pad : bool, optional
        allow zero-padding of signal to nearest power of 2 or 3 in order
        to speed up computation

    Returns
    -------
    envelope : 1D array

    """
    # filter
    if filter_options is None:
        filter_options = {}
    if smooth_options is None:
        smooth_options = {}
    if not isfiltered:
        if freq_band is None:
            raise ValueError("Please specify frequency band")
        filter_arg = dict(transition_width="25%", attenuation=60)
        filter_arg.update(filter_options)
        envelope = apply_filter(signals, freq_band, axis=axis, fs=fs, **filter_arg)
    else:
        envelope = signals

    # compute envelope
    if not isinstance(envelope, (tuple, list)):
        envelope = [
            envelope,
        ]

    if len(envelope) == 0:
        raise ValueError("No signal provided.")

    # check that all arrays in the list have the same size along axis
    if not all([x.shape[axis] == envelope[0].shape[axis] for x in envelope]):
        raise ValueError("Signals in list do not have compatible shapes")

    n = envelope[0].shape[axis]
    if pad:
        norig = n
        n = int(np.min([2, 3] ** np.ceil(np.log(n) / np.log([2, 3]))))

    envelope = cast(list, envelope)
    for k in range(len(envelope)):
        envelope[k] = np.abs(sp.signal.hilbert(envelope[k], N=n, axis=axis))
        if envelope[k].ndim > 1:
            envelope[k] = np.mean(
                np.rollaxis(envelope[k], axis).reshape(
                    [
                        envelope[k].shape[axis],
                        int(envelope[k].size / envelope[k].shape[axis]),
                    ]
                ),
                axis=1,
            )

    if len(envelope) > 1:
        envelope = reduce(np.add, envelope) / len(envelope)
    else:
        envelope = envelope[0]

    if pad:
        envelope = envelope[:norig]

    # (optional) smooth envelope
    smooth_arg = dict(kernel="gaussian", bandwidth=-1.0)
    smooth_arg.update(smooth_options)
    if smooth_arg["bandwidth"] > 0:
        envelope = smooth1d(envelope, delta=1.0 / fs, **smooth_arg)

    return envelope


def segment_contains(segment, x, issorted=True, expand=None):
    """Test if values are contained in segments.

    Segments are considered left closed and right open intervals. So,
    a value x is contained in a segment if start<=x and x<stop.

    Parameters
    ----------
    segment : segment array
    x : 1d array
    issorted : bool
        Assumes vector `x` is sorted and will not sort it internally.
        Note that even if `issorted` is False, the third output argument
        will still return indices into the (internally) sorted vector.
    expand : bool
        Will expand the last output to full index arrays into 'x' for
        each segment. The default is True if `issorted` is False and
        vice versa. Note that for non-sorted data (`issorted` is False) and
        `expand`=False, the last output argument will contain start and stop
        indices into the (internally) sorted input array.

    Returns
    -------
    ndarray
        True for each value in `x` that is contained within any segment.
    ndarray
        For each segment the number of values in `x` that it contains.
    ndarray
        For each segment, the start and end indices of values in SORTED
        vector `x` that are contained within that segment.

    """

    x = np.array(x).ravel()

    if expand is None:
        expand = not issorted

    if not issorted:
        sort_indices = np.argsort(x)
        x = x[sort_indices]

    segment = check_segments(segment, copy=False)
    nseg = segment.shape[0]
    nx = len(x)

    xp = 0  # index of current x value
    xfillp = 0  # index of last x value that has been tested
    # True for each x inside any of the segments
    isinseg = np.zeros(x.shape, dtype=np.bool)
    # for each segment start and end index of x-values
    # that are contained within the segment
    contains = -1 * np.ones(segment.shape, dtype=int)
    ninseg = np.zeros(
        segment.shape[0], dtype=int
    )  # for each segment number of x-values it contains

    if nx > 0:
        # loop through all segments
        for sp in range(nseg):
            if x[xp] < segment[sp, 0]:  # current x is before segment start
                # find first x inside segment
                idx = np.searchsorted(
                    x[xp:], segment[sp, 0], side="left"
                )  # switch to 'right' to make left open interval
                if (idx + xp) >= nx:  # no x inside segment found
                    break
                if (
                    x[idx + xp] >= segment[sp, 1]
                ):  # Use '>' to make right closed interval
                    continue
                xp += idx  # update current x
                contains[sp, 0] = xp  # mark first x for this segment
            elif x[xp] >= segment[sp, 1]:
                # x-value is past current segment, # go to next segment.
                # Use '>' to make right closed interval
                continue
            else:
                contains[sp, 0] = xp  # mark first x for this segment

            # find last x in segment and mark it
            xlastp = (
                xp + np.searchsorted(x[xp:], segment[sp, 1], side="left") - 1
            )  # switch to 'right' to make right closed interval
            contains[sp, 1] = xlastp

            # count number of x values in segment
            ninseg[sp] = contains[sp, 1] - contains[sp, 0] + 1

            # fast forward current fill index if needed
            if xp > xfillp:
                xfillp = xp

            # mark x-values as contained
            isinseg[xfillp : (xlastp + 1)] = 1

            # set new current fill index
            if xfillp < xlastp:
                xfillp = xlastp

    if not issorted:
        isinseg = isinseg[np.argsort(sort_indices)]

    if expand:
        if issorted:
            contains = [np.arange(start, stop + 1) for start, stop in contains]
        else:
            contains = [sort_indices[start : stop + 1] for start, stop in contains]

    return isinseg, ninseg, contains


def check_segments(x: np.ndarray, copy=False):
    """Convert to segment array.

    Parameters
    ----------
    x : 1d array-like or (n,2) array-like
    copy : bool
        the output will always be a copy of the input

    Returns
    -------
    (n,2) array

    """

    try:
        if copy is False:
            try:
                x = np.asarray(x, copy=False)
            except ValueError:
                x = np.asarray(x)  # Fallback: copy if needed
        else:
            x = np.asarray(x, copy=True)
    except TypeError:
        raise ValueError("Cannot convert data to numpy array") from None

    # The array needs to contain real values only.
    # Is this a proper general test for numbers?
    if not np.isrealobj(x):
        raise ValueError("Values are not real numbers")

    # The array has to have two dimensions of shape(X,2), where X>=0.
    # As a special case, a one dimensional vector of at least length two is considered
    # a valid list of segments, e.g. when data is specified as a list [0,2,3].

    if x.shape == (0,):
        x = np.zeros([0, 2])
    elif x.ndim == 1 and len(x) > 1:
        x = np.vstack((x[0:-1], x[1:])).T
    elif x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Incorrect array size")

    # Negative duration segments are not allowed.
    if np.any(np.diff(x, axis=1) < 0):
        raise ValueError("Segment durations cannot be negative")

    return x


def ripple_envelope(signals, band="ripple", **kwargs):

    smooth_options = dict(kernel="gaussian", bandwidth=0.0075)
    smooth_options.update(kwargs.pop("smooth_options", {}))

    filter_options = dict(transition_width="25%", attenuation=60)
    filter_options.update(kwargs.pop("filter_options", {}))

    return compute_envelope(
        signals,
        band,
        filter_options=filter_options,
        smooth_options=smooth_options,
        **kwargs,
    )


def detect_ripples(
    time: npt.NDArray[np.float64],
    signals: npt.NDArray[np.float64],
    axis: int = -1,
    band: str | Sequence[float] = "ripple",
    isenvelope: bool = False,
    isfiltered: bool = False,
    segments: Any = None,
    threshold: float
    | Sequence[float]
    | Callable[[npt.NDArray[np.float64]], Sequence[float]]
    | None = None,
    threshold_dev: Sequence[float] | None = None,
    allowable_gap: float = 0.02,
    minimum_duration: float = 0.03,
    filter_options: dict[str, Any] | None = None,
    smooth_options: dict[str, Any] | None = None,
) -> list[RippleEvent]:
    """Detect high-frequency ripple events in multi-channel data streams.

    Args:
        time: 1D array of timestamps in seconds.
        signals: Raw, filtered, or pre-computed envelope signal array.
        axis: Time dimension index in the signals array.
        band: Target frequency band string or [low, high] sequence.
        isenvelope: True if signals is already a pre-computed envelope.
        isfiltered: True if signals is already filtered.
        segments: Array-like constraints to restrict detection boundaries.
        threshold: Absolute value cutoffs or dynamic callable threshold generator.
        threshold_dev: Median-based deviation scaling parameters [low, high].
        allowable_gap: Max gap in seconds to merge adjacent events.
        minimum_duration: Min duration in seconds to keep an event.
        filter_options: Backend parameters passed to the signal filter.
        smooth_options: Backend parameters passed to the envelope smoother.
        device_start_us: Baseline DAQ hardware clock timestamp in microseconds.

    Returns:
        List of validated RippleEvent Pydantic models.
    """
    opts_smooth = smooth_options or {}
    opts_filter = filter_options or {}
    dt = float(np.median(np.diff(time)))

    if not isenvelope:
        envelope = ripple_envelope(
            signals,
            band=band,
            axis=axis,
            fs=1.0 / dt,
            isfiltered=isfiltered,
            filter_options=opts_filter,
            smooth_options=opts_smooth,
        )
    else:
        envelope = signals
        if envelope.ndim != 1:
            raise ValueError("Envelope needs to be a 1D vector.")

    search_segments = check_segments(
        segments if segments is not None else [-np.inf, np.inf]
    )
    in_segment, _, _ = segment_contains(search_segments, time)

    if threshold is None and threshold_dev is None:
        mean = float(np.mean(envelope[in_segment]))
        dev = float(np.std(envelope[in_segment]))
        calc_threshold = [mean + dev, mean + 4.0 * dev]
    elif threshold_dev is not None:
        median = float(np.median(envelope))
        dev = median - float(envelope.min())
        calc_threshold = (dev * np.array(threshold_dev) + median).tolist()
    elif callable(threshold):
        calc_threshold = list(threshold(envelope[in_segment]))
    else:
        calc_threshold = np.asanyarray(threshold, dtype=np.float64).ravel().tolist()

    low = float(calc_threshold[0])
    high = float(calc_threshold[-1])

    ripple_peak_time, _ = localmaxima(
        envelope, x=time, method="gradient", yrange=[high, np.inf]
    )

    ripple_segments = detect_mountains(
        envelope, x=time, low=low, high=high, segments=search_segments
    )

    ripple_segments.ijoin(gap=allowable_gap)
    del ripple_segments[ripple_segments.duration < minimum_duration]

    events: list[RippleEvent] = []
    for start, end in zip(
        ripple_segments.start,
        ripple_segments.stop,
        strict=False,
    ):
        mask = (ripple_peak_time >= start) & (ripple_peak_time <= end)
        matching_peaks = ripple_peak_time[mask]

        if matching_peaks.size > 0:
            peak = float(matching_peaks[0])
            events.append(
                RippleEvent(
                    start_sec=float(start),
                    end_sec=float(end),
                    peak_sec=peak,
                )
            )

    return events
