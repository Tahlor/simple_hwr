import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
import sys
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pylab as plt
import math
from PIL import ImageEnhance, Image

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}
cv2.setNumThreads(0)

def change_contrast(img, min_contrast=.25, max_contrast=1.3, contrast=None):
    if isinstance(img, np.ndarray):
        if img.ndim > 2:
            assert img.shape[-1]==1
            img = img[:, :, 0]
        img = Image.fromarray(np.uint8(img), "L")
    enhancer = ImageEnhance.Contrast(img)
    if contrast is None:
        contrast = np.random.rand()*(max_contrast-min_contrast)+min_contrast
    #Image.fromarray(np.array(enhancer.enhance(contrast))).show()
    return np.array(enhancer.enhance(contrast))

def occlude(img, occlusion_size=1, occlusion_freq=.5, occlusion_level=1, logger=None, noise_type=None):
    if occlusion_freq:
        return _occlude(img, occlusion_size, occlusion_freq, occlusion_level, logger)
    else:
        if noise_type is None:
            noise_type = "gaussian"
        return noise(img, occlusion_level=occlusion_level, logger=logger, noise_type=noise_type)

def _occlude(img, occlusion_size=1, occlusion_freq=.5, occlusion_level=1, logger=None):
    """
        Occlusion frequency : between 0% and this number will be occluded
        Occlusion level: maximum occlusion change (multiplier); each pixel to be occluded has a random occlusion probability;
                         then it is multiplied/divided by at most the occlusion level

        NOT IMPLEMENTED:
        OTHER OPTIONS:
            RANDOM OCCLUSION THRESHOLD
            RANDOM OCCLUSION LEVEL (within range)
            OCCLUSION SIZE
    Args:
        img:
        occlusion_size:
        occlusion_freq:
        occlusion_level: just "dim" these pixels a random amount; 1 - white, 0 - original image
        occlusion
        logger:

    Returns:

    """
    # Randomly choose occlusion frequency between 0 and specified occlusion
    # H X W X Channel
    random_state = np.random.RandomState()
    occlusion_freq = random_state.uniform(0, occlusion_freq) #
    binary_mask = random_state.choice(2, img.shape, p=[occlusion_freq, 1-occlusion_freq])
    #logger.debug(binary_mask)
    if occlusion_level==1:
        occlusion = np.where(binary_mask==0, 255, img) # replace 0's with white
    else: # 1 = .3
        sd = occlusion_level / 2 # ~95% of observations will be less extreme; if max_intensity=1, we set so 95% of multipliers are <1
        random_mask = random_state.randn(*img.shape, ) * sd # * 2 - max_intensity # min -occlusion, max occlusion
        random_mask = np.clip(random_mask, -1, 1)
        if False: # randomly whiten to different levels
            occlusion = np.where(binary_mask == 0, (1-random_mask)*img+255*random_mask, img)  # replace 0's with white
        else: # random noise
            random_mask = np.minimum((random_mask + 1) * img, 255)
            occlusion = np.where(binary_mask == 0, random_mask, img)
    return occlusion

# def noise(img, occlusion_size=1, occlusion_freq=.5, max_intensity=1, logger=None):
#     """
#         NOT IMPLEMENTED:
#         OTHER OPTIONS:
#             RANDOM OCCLUSION THRESHOLD
#             RANDOM OCCLUSION LEVEL (within range)
#             OCCLUSION SIZE
#     Args:
#         img:
#         occlusion_size:
#         occlusion_freq:
#         max_intensity: just "dim" these pixels a random amount; 1 - white, 0 - original image
#         occlusion
#         logger:
#
#     Returns:
#
#     """
#
#
#     # Randomly choose occlusion frequency between 0 and specified occlusion
#     # H X W X Channel
#     random_state = np.random.RandomState()
#     occlusion_freq = random_state.uniform(0, occlusion_freq)
#     binary_mask = random_state.choice(2, img.shape, p=[occlusion_freq, 1-occlusion_freq])
#     #logger.debug(binary_mask)
#     if max_intensity==1:
#         occlusion = np.where(binary_mask==0, 255, img) # replace 0's with white
#     else:
#         random_mask = random_state.rand(*img.shape) * max_intensity # occlude between not-at-all and occlusion-level
#         occlusion = np.where(binary_mask == 0, (1-random_mask)*img+255*random_mask, img)  # replace 0's with white
#
#     return occlusion

def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 25)
    w_mesh_std = kwargs.get('w_mesh_std', 3.0)

    h_mesh_interval = kwargs.get('h_mesh_interval', 25)
    h_mesh_std = kwargs.get('h_mesh_std', 3.0)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        #############################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2 or img.shape[2]==1: # if already grayscale
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(255,255,255))

    return warped

def warp_points(points, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 25)
    w_mesh_std = kwargs.get('w_mesh_std', 3.0)

    h_mesh_interval = kwargs.get('h_mesh_interval', 25)
    h_mesh_std = kwargs.get('h_mesh_std', 3.0)

    interpolation_method = kwargs.get('interpolation', 'linear')

    # recenter and pad in here
    padding_h = int(h_mesh_std * 3)# 3 stdev from the mean, both directions
    padding_w = int(w_mesh_std * 3)
    bounds = np.amax(points, axis=0) + 1
    h, w = bounds[0] + padding_h*2, bounds[1] + padding_w*2
    points[:, 0] += padding_h
    points[:, 1] += padding_w

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] += np.clip(random_state.normal(0.0, h_mesh_std, size=source_shape), -h_mesh_std, h_mesh_std)
    destination[:,1] += np.clip(random_state.normal(0.0, w_mesh_std, size=source_shape), -w_mesh_std, w_mesh_std)

    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]

    def get_points(pt):
        t, b = math.ceil(pt[0]), math.floor(pt[0])
        l, r = math.ceil(pt[1]), math.floor(pt[1])
        return np.mean((map_y[t][l], map_y[t][r], map_y[b][l], map_y[b][r])), np.mean((map_x[t][l], map_x[t][r], map_x[b][l], map_x[b][r]))

    warped = np.array([get_points(pt) for pt in points]) # WIDTH x 2
    warped[:, 0:2] -= np.min(warped)
    warped[:, 0] *= bounds[0]/(np.max(warped[:, 0]) + 1)
    warped[:, 1] *= bounds[1]/(np.max(warped[:, 1]) + 1)
    return warped

def noise(img, occlusion_level=1, logger=None, noise_type="gaussian"):
    if noise_type == "gaussian":
        return gaussian_noise(img, max_intensity=occlusion_level, logger=logger)
    else:
        raise Exception("Not implemented")

def crop(img, threshold=200, padding=10):
    all_ink = np.where(img < threshold)
    try:
        first_ink = max(0, np.min(all_ink[1]) - padding)
        last_ink = min(np.max(all_ink[1])+padding, img.shape[1])

        # Must be at least 50 pixels wide
        if last_ink - first_ink > 50:
            return img[:, first_ink:last_ink]
        else:
            return img
    except:
        return img

# https://github.com/tmbdev/das2018-tutorial/blob/master/40-augmentation.ipynb
def random_distortions(img, sigma=20.0, noise_max=10.0):
    n, m = img.shape

    sigma = np.random.uniform(8, sigma)
    noise_max = np.random.uniform(1, noise_max)

    noise = np.random.rand(2, n, m)
    noise = ndimage.gaussian_filter(noise, (0, sigma, sigma))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    noise = (2*noise-1) * noise_max

    assert noise.shape[0] == 2
    assert img.shape == noise.shape[1:], (img.shape, noise.shape)

    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    noise += xy
    distorted = ndimage.map_coordinates(img, noise, order=1, mode="reflect")
    return distorted

def blur(img, max_intensity=1.5):
    max_intensity = np.random.uniform(0, max_intensity)
    return ndimage.gaussian_filter(img, max_intensity)

def gaussian_noise(img, max_intensity=.1, logger=None):
    """
        Expects images on 0-255 scale
        max_intensity: .1 - light haze, 1 heavy

        Adds random noise to image
    """

    random_state = np.random.RandomState()
    sd = min(abs(np.random.normal()) * max_intensity / 2, max_intensity / 2)
    #sd = max_intensity / 2  # ~95% of observations will be less extreme; if max_intensity=1, we set so 95% of multipliers are <1
    noise_mask = random_state.randn(*img.shape, ) * sd  # * 2 - max_intensity # min -occlusion, max occlusion
    noise_mask = np.clip(noise_mask, -1, 1) * 255/2
    noisy_img = np.clip(img + noise_mask, 0, 255)
    return noisy_img

    # elif noise_typ == "s&p":
    #     row, col, ch = image.shape
    #     s_vs_p = 0.5
    #     amount = 0.004
    #     out = image
    #     # Salt mode
    #     num_salt = np.ceil(amount * image.size * s_vs_p)
    #     coords = [np.random.randint(0, i - 1, int(num_salt))
    #               for i in image.shape]
    #     out[coords] = 1
    #
    #     # Pepper mode
    #     num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #     coords = [np.random.randint(0, i - 1, int(num_pepper))
    #               for i in image.shape]
    #     out[coords] = 0
    #     return out
    # elif noise_typ == "poisson":
    #     vals = len(np.unique(image))
    #     vals = 2 ** np.ceil(np.log2(vals))
    #     noisy = np.random.poisson(image * vals) / float(vals)
    #     return noisy
    # elif noise_typ == "speckle":
    #     row, col, ch = image.shape
    #     gauss = np.random.randn(row, col, ch)
    #     gauss = gauss.reshape(row, col, ch)
    #     noisy = image + image * gauss
    #     return noisy

def elastic_transform(image, alpha=2.5, sigma=1.1, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    alpha = np.random.uniform(1,alpha)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, cval=255).reshape(shape)

def get_test_image():
    input_image = "data/prepare_IAM_Lines/lines/m04/m04-061/m04-061-02.png"
    input_image = "../data/sample_offline/a05-039-00.png"
    #input_image = "data/sample_online/0_6cfd6616717146a687391b52621340c1.tif"
    img = cv2.imread(input_image, 0)
    plot(img, "Original image")
    return img

def plot(img, title):
    plt.figure(dpi=400)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.show()

def test():
    if False:
        input_image = sys.argv[1]
        output_image = sys.argv[2]
        img = cv2.imread(input_image)
        cv2.imwrite(output_image, img)
    else:
        img = get_test_image()
        noisy = noise(img, occlusion_level=.5)
        plot(noisy, "With Noise")

        distorted = random_distortions(img)
        plot(distorted, "With distortion")

        blurred = blur(img)
        plot(blurred, "With blur")

        distorted2 = elastic_transform(img,  alpha=3, sigma=1.1)
        plot(distorted2, "Distorted2")

def test_crop():
        img = get_test_image()
        cropped = crop(img)
        plt.imshow(cropped, cmap="gray")
        plt.title("cropped")
        plt.show()

def test_wavy_distortion(img):
    for i in range(0,5):
        distorted = random_distortions(img)
        plot(distorted, "With distortion")

def test_blur(img):
    for i in range(0,5):
        blurred = blur(img)
        plot(blurred, "With blur")

def test_gaussian(img):
    for i in range(0,5):
        gauss = change_contrast(gaussian_noise(img, max_intensity=.4, logger=None), contrast=(i+1)*.1)
        plot(gauss, "gauss")

def test_contrast(img):
    for i in range(0,5):
        contrast = change_contrast(img, contrast=(i+1)*.2)
        plot(contrast, "contrast")

if __name__ == "__main__":
    img = get_test_image()
    test_contrast(img)

    #test_wavy_distortion(img)
    # img = get_test_image()
    #
    # distorted2 = elastic_transform(img, alpha=1, sigma=1.1)
    # plot(distorted2, "Distorted2")


