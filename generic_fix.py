from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
import sys
import numpy as np

sys.path.append('../d2/data/tub_1_17-10-23/')

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

good_image = Image.open('../d2/data/tub_1_17-10-23/136_cam-image_array_.jpg')
good_array = np.asarray(good_image)

#Original Image
im = Image.open('../d2/data/tub_1_17-10-23/6_cam-image_array_.jpg')
im.show()

#Equalizing Histogram
im2 = ImageOps.equalize(im)
im2.show()

#Equalizing Histogram
contrast = ImageEnhance.Contrast(im2)
#contrast.enhance(2).show()

im_array = np.asarray(im)
im2_array = np.asarray(im2)

matched_array = hist_match(im_array, good_array)
im3 = Image.fromarray(np.uint8(matched_array))
#im3.show()
