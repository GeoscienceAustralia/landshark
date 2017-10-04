"""Importing routines for tif data."""

import os.path
import logging
from collections import namedtuple

import rasterio
import numpy as np
from typing import Callable, Any, List, Tuple, Iterator, Union
from mypy_extensions import NoReturn
from rasterio._io import RasterReader

from landshark.image import pixel_coordinates


log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]

# Convenience types
Band = namedtuple("Band", ["image", "index"])
BandCollection = namedtuple("BandCollection", ["ordinal", "categorical"])


class ImageStack:
    """A stack of registered images with the same res and bbox.

    This class simplifies the handling of multiple geotiff files
    in order to emulate the behaviour of a single many-banded image.

    Parameters
    ----------
    path_list : List[str]
        The paths to image files that will comprise the stack.
    block_rows : Union[None, int]
        Optional integer > 0 that specifies the number of rows read at a time.
        If not provided then a semi-sensible value is computed.

    """

    def __init__(self, path_list: List[str],
                 block_rows: Union[None, int] = None) -> None:
        """Construct an instance of ImageStack."""
        images = [rasterio.open(k, "r") for k in path_list]
        width = _match(lambda x: x.width, images, "width")
        height = _match(lambda x: x.height, images, "height")
        affine = _match(lambda x: x.affine, images, "affine")
        coords_x, coords_y = pixel_coordinates(width, height, affine)
        # crs = _match(lambda x: str(x.crs.data), images, "crs")
        # TODO affine transform
        bands = _bands(images)
        ordinal_bands = bands.ordinal
        categorical_bands = bands.categorical
        ordinal_names = _names(ordinal_bands)
        categorical_names = _names(categorical_bands)
        ordinal_dtype = np.float32
        categorical_dtype = np.int32
        ordinal_missing = _missing(ordinal_bands, dtype=ordinal_dtype)
        categorical_missing = _missing(categorical_bands,
                                       dtype=categorical_dtype)
        if not block_rows:
            block_rows = max(_block_rows(ordinal_bands),
                             _block_rows(categorical_bands))
            log.info("Using tif block size of {} rows".format(block_rows))
        else:
            log.info("User set tif block size of {} rows".format(block_rows))
        windows = _windows(width, height, block_rows)
        log.info("Found {} ordinal bands".format(len(ordinal_bands)))
        log.info("Found {} categorical bands".format(len(categorical_bands)))
        log.info("Image resolution is {} x {}".format(width, height))

        self.max_block_size = block_rows * width
        self.categorical_dtype = categorical_dtype
        self.ordinal_dtype = ordinal_dtype
        self.width = width
        self.height = height
        self.affine = affine
        self.coordinates_x = coords_x
        self.coordinates_y = coords_y
        self.block_rows = block_rows
        self.windows = windows
        self.ordinal_bands = ordinal_bands
        self.categorical_bands = categorical_bands
        self.ordinal_names = ordinal_names
        self.categorical_names = categorical_names
        self.ordinal_missing = ordinal_missing
        self.categorical_missing = categorical_missing

    def categorical_blocks(self) -> Iterator[np.ndarray]:
        """
        Create an iterator over categorical blocks from the image stack.

        The iterator will pass through the data only once.

        Returns
        -------
        gen : Iterator[np.ndarray]
            An iterator that has categorical (int-valued) image segments
            with block_rows rows of data from the image stack.

        """
        gen = _read(self.categorical_bands, self.windows,
                    self.categorical_dtype)
        return gen

    def ordinal_blocks(self) -> Iterator[np.ndarray]:
        """
        Create an iterator over ordinal blocks from the image stack.

        The iterator will pass through the data only once.

        Returns
        -------
        gen : Iterator[np.ndarray]
            An iterator that has ordinal (float-valued) image segments
            with block_rows rows of data from the image stack.

        """
        gen = _read(self.ordinal_bands, self.windows,
                    self.ordinal_dtype)
        return gen


def _match(f: Callable[[Any], Any],
           images: List[RasterReader],
           name: str) -> Any:
    """Return specified property of images if they match."""
    property_list = [f(k) for k in images]
    property_set = set(property_list)
    if len(property_set) != 1:
        _fatal_mismatch(property_list, images, name)
    result = property_set.pop()
    return result


def _fatal_mismatch(property_list: List[Any],
                    images: List[RasterReader],
                    name: str) -> NoReturn:
    """Print a fatal log with helpful table of property mismatch."""
    assert len(property_list) == len(images)

    image_names = [os.path.basename(k.name) for k in images]
    props = zip(image_names, property_list)
    strings = ["{}: {}".format(i, l) for i, l in props]
    table = "\n".join(strings)
    log.error("No match for {}:\n{}".format(name, table))
    raise ValueError("No match for input image property {}".format(name))


def _names(band_list: List[Band]) -> List[str]:
    """Generate a list of band names."""
    band_names = []
    for im, band_idx in band_list:
        basename = os.path.basename(im.name)
        if im.count > 1:
            name = basename + "_{}".format(band_idx)
        else:
            name = basename
        band_names.append(name)
    return band_names


def _missing(bands: List[Band], dtype: np.dtype) -> List[Any]:
    """
    Convert missing data values to a given dtype (rasterio workaround).

    Note that the list may contain 'None' where there are no missing values.
    """
    l = [b.image.nodatavals[b.index - 1] for b in bands]

    def convert(x: Union[None, float]) -> Any:
        return dtype(x) if x is not None else x
    r = [convert(k) for k in l]
    return r


def _bands(images: List[RasterReader]) -> BandCollection:
    """Get list of ordinaal and categorical bands from list of images."""
    # Dont use np.int32 etc here for weird comparison breakage
    categ_types = ["int32", "int64", "uint8", "uint32", "uint64", "uint8"]
    categ_set = {np.dtype(k) for k in categ_types}
    ordin = []
    categ = []
    for im in images:
        for i, d in enumerate(im.dtypes):
            band = Band(image=im, index=(i + 1))   # bands start from 1
            if np.dtype(d) in categ_set:
                categ.append(band)
            else:
                ordin.append(band)

    result = BandCollection(ordinal=ordin, categorical=categ)
    return result


def _block_rows(bands: List[Band]) -> int:
    """Choose a sensible (global) blocksize based on input images' blocks."""
    block_list = []
    for b in bands:
        block = b.image.block_shapes[b.index - 1]
        if not block[0] <= block[1]:
            raise ValueError("No support for column-wise blocks")
        block_list.append(block[0])
    blockrows = int(np.amax(block_list))  # LCM would be more efficient but meh
    return blockrows


def _windows(width: int, height: int,
             block_rows: int) -> List[WindowType]:
    """Create a list of windows to cover the image."""
    assert width > 0
    assert height > 0
    assert block_rows > 0
    n = height // block_rows
    ret = [((i * block_rows, (i + 1) * block_rows), (0, width))
           for i in range(n)]
    if height % block_rows != 0:
        final_window = ((n * block_rows, height), (0, width))
        ret.append(final_window)
    return ret


def _block_shape(window: WindowType, nbands: int) -> Tuple[int, int, int]:
    """Compute the shape of the output of iterators from the window size."""
    rows = window[0][1] - window[0][0]
    cols = window[1][1] - window[1][0]
    result = (rows, cols, nbands)
    return result


def _read(band_list: List[Band], windows: List[WindowType],
          dtype: np.dtype) -> Iterator[np.ndarray]:
    """Create a generator that yields blocks of the image stack."""
    for w in windows:
        block_shape = _block_shape(w, len(band_list))
        out_array = np.empty(block_shape, dtype=dtype)
        for i, b in enumerate(band_list):
            a = b.image.read(b.index, window=w)
            out_array[:, :, i] = a.astype(dtype)
        yield out_array