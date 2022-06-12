from PIL import Image
import io, os

from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
from torchvision import datasets,transforms

import skimage.io
import matplotlib.pyplot as plt
import skimage.filters
from skimage import color
import numpy as np
from methods.load_process_imgnet_val_data import  load_imgnet_val_data

from scipy.stats import entropy
from math import log, e
import pandas as pd

import numpy.ma as ma
import timeit

def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

def entropy3(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()

def entropy4(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()



def convert_to_webp(source):
    """Convert image to webp.

    Args:
        source (pathlib.Path): Path to source image

    Returns:
        pathlib.Path: path to new image
    """
    # destination = source.with_suffix(".webp")
    destination = io.BytesIO()
    destination2 = io.BytesIO()
    image = Image.open(source)  # Open image
    print(type(image))
    image.save(destination, format="webp")  # Convert image to webp
    image.save(destination2, format="png")  # Convert image to webp
    print('File size using io.BytesIO : ', destination.tell())
    print('File size using io.BytesIO : ', destination2.tell())

    return destination


def main():
    paths = Path("images").glob("**/*.png")
    for path in paths:
        webp_path = convert_to_webp(path)
        # print(webp_path)

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

def add_random_pixels(img, p=0.01):

    dim = img.shape[0]
    no_of_ones = int(dim * dim * p)
    a = np.zeros(dim * dim - no_of_ones).astype(int)
    b = np.ones(no_of_ones).astype(int)
    mask = np.concatenate([a, b], axis=0)
    np.random.seed(0)
    mask = np.random.permutation(mask).reshape(dim, dim)
    mask_3d = np.stack((mask, mask, mask), axis=2)
    random_img = np.where(mask_3d == 1, img, 0)

    return random_img, mask

def calculate_webp_size(numpy_image):
    image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    out_original = io.BytesIO()
    image.save(out_original, format = "webp")
    im_size = out_original.tell()
    return im_size

def calculate_relative_entropy(main_img_size, full_blurred_size, saliency_img_size):
    return (saliency_img_size - full_blurred_size)/(main_img_size - full_blurred_size)

def interpolate_img(img, mask, interp_mode='linear'):
    img_copy = img.copy()
    img_copy[:, :, 0] = interpolate_missing_pixels(img_copy[:, :, 0], mask, method=interp_mode)
    img_copy[:, :, 1] = interpolate_missing_pixels(img_copy[:, :, 1], mask, method=interp_mode)
    img_copy[:, :, 2] = interpolate_missing_pixels(img_copy[:, :, 2], mask, method=interp_mode)
    return img_copy

def get_thresholded_saliency_mask_tensor(saliency_mask, threshold_percent):
    
    '''
    Input: Saliency Mask and x% threshold 
    Output: Corresponding saliency mask
    '''
    topk_values = torch.topk(saliency_mask.view(-1), int(saliency_mask.view(-1).shape[0]*threshold_percent))
    
    feature_ranking = torch.zeros_like(saliency_mask.view(-1))
    
    feature_ranking[topk_values[1]] = 1
    
    thresholded_saliency_mask = feature_ranking.reshape(saliency_mask.shape)
       
    return thresholded_saliency_mask

def get_thresholded_saliency_mask_numpy(saliency_mask, threshold_percent):
    
    thresholded_saliency_mask = np.zeros_like(saliency_mask.flatten())
    no_of_required_salient_values = int(saliency_mask.flatten().shape[0]*threshold_percent)
    
    if no_of_required_salient_values > 0:
        topk_salient_indices = np.argpartition(saliency_mask.flatten(), \
                                           -no_of_required_salient_values)[-no_of_required_salient_values:]
        thresholded_saliency_mask[topk_salient_indices] = 1
    
    thresholded_saliency_mask = thresholded_saliency_mask.reshape(saliency_mask.shape)
    
    return thresholded_saliency_mask

def generate_img_mask(saliency_mask, img_dim, percent, mask_rand):
    if saliency_mask is None:
        
        no_of_ones = int(img_dim*img_dim*percent)
        a = np.zeros(img_dim*img_dim - no_of_ones).astype(int)
        b = np.ones(no_of_ones).astype(int)
        mask = np.concatenate([a, b], axis=0)
        mask = np.random.permutation(mask).reshape(img_dim, img_dim)

    else:
        mask = get_thresholded_saliency_mask_numpy(saliency_mask, percent)

    # create 0/1 mask
    # if p = 0.0, use only the first 1% random mask, otherwise use the mask as specified above
    mask = 1 - (1*mask_rand) if percent == 0.0 else mask

    mask_3d = np.stack((mask,mask,mask),axis=2)
    
    return mask_3d

def calculate_information_size(img):
    
    information_size = entropy1(color.rgb2gray(img))
    
    return information_size
    

def generate_saliency_focused_images_prev(numpy_image, rand_img, mask_rand, main_size, p_start=0.0, p_end=1.0, iterations=50, saliency_mask=None, interp_mode = 'linear'):

    """Generate saliency-focused images for different saliency thresholds"""

    img_pairs = []
    pixels, normalized_entropy = [], []
    dim = numpy_image.shape[0]
    full_blurred_size = calculate_webp_size(interpolate_img(rand_img, mask_rand, interp_mode=interp_mode))
    
    print('Ensuring usage of initial random mask')
    initial_random_mask = 1 - (1*mask_rand)

    # Add %x salient pixels

    for p in np.linspace(p_start, p_end, num=iterations, endpoint=True):

        pixels.append(p)

        if saliency_mask is None:
            no_of_ones = int(dim*dim*p)
            a = np.zeros(dim*dim - no_of_ones).astype(int)
            b = np.ones(no_of_ones).astype(int)
            new_mask = np.concatenate([a, b], axis=0)
            new_mask = np.random.permutation(new_mask).reshape(dim, dim)

        else:
            if p == 0.0:
                print('Using Real Sal Mask')
            new_mask = get_thresholded_saliency_mask_numpy(saliency_mask, p)

        # create 0/1 mask
        
        mask = 1*((new_mask == 1) | (initial_random_mask == 1))                

        mask_3d = np.stack((mask,mask,mask),axis=2)

        saliency_img = np.where(mask_3d==1, numpy_image, rand_img)
        saliency_img_copy = saliency_img.copy()

        # create boolean mask

        mask = ma.make_mask(1 - mask)
                  
        if p < 1.0:
            saliency_img_copy[:, :, 0] = interpolate_missing_pixels(saliency_img_copy[:, :, 0], mask, method=interp_mode)
            saliency_img_copy[:, :, 1] = interpolate_missing_pixels(saliency_img_copy[:, :, 1], mask, method=interp_mode)
            saliency_img_copy[:, :, 2] = interpolate_missing_pixels(saliency_img_copy[:, :, 2], mask, method=interp_mode)

        img_pairs.append([saliency_img, saliency_img_copy])
        
        saliency_img_size = calculate_webp_size(saliency_img_copy)
        relative_entropy = calculate_relative_entropy(main_size, full_blurred_size, saliency_img_size)

        print(f'T (%):{p:.3f} \t S.Ratio: {saliency_img_size:.2f}/{main_size:.2f} \t RE: {saliency_img_size/main_size:.3f} \t Rel.E: {relative_entropy:.3f}')
        normalized_entropy.append(relative_entropy)
    
    normalized_entropy = np.array(normalized_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0, 1)
    percent_entropy = np.stack([pixels, normalized_entropy], axis=0)

    img_pairs = np.stack(img_pairs, axis=0)

    return img_pairs, percent_entropy

def generate_revised_saliency_focused_images(numpy_image, rand_img, mask_rand, rand_interpolated_img, iterations=50, saliency_mask=None, interp_mode='linear'):

    """
    Generate saliency-focused images for different saliency thresholds
    No interpolation after adding important pixels 
    
    """

    saliency_all_images = []
    pixels, normalized_entropy = [], []
    dim = numpy_image.shape[0]
    main_size = calculate_information_size(numpy_image)
    full_blurred_size = calculate_information_size(rand_img)
    
    print('Ensuring usage of initial random mask')
    initial_random_mask = 1 - (1*mask_rand)

    # Add %x salient pixels
    
    p_start, p_end = 0.0, 1.0

    for p in np.linspace(p_start, p_end, num=iterations, endpoint=True):

        pixels.append(p)

        if saliency_mask is None:
            no_of_ones = int(dim*dim*p)
            a = np.zeros(dim*dim - no_of_ones).astype(int)
            b = np.ones(no_of_ones).astype(int)
            new_mask = np.concatenate([a, b], axis=0)
            new_mask = np.random.permutation(new_mask).reshape(dim, dim)

        else:
            if p == 0.0:
                print('Using Real Sal Mask')
            new_mask = get_thresholded_saliency_mask_numpy(saliency_mask, p)

        # create 0/1 mask  
        
        mask = 1*((new_mask == 1) | (initial_random_mask == 1))                
        mask_3d = np.stack((mask,mask,mask),axis=2)

    
        saliency_img = np.where(mask_3d==1, numpy_image, rand_interpolated_img)
        saliency_img_only = mask_3d*numpy_image  # No interpolation on the updates
#         saliency_img_filled = np.where(mask_3d==1, numpy_image, int(np.mean(numpy_image)))
       
        saliency_all_images.append([saliency_img_only, saliency_img])
        
        saliency_img_size = calculate_information_size(saliency_img_only)
        relative_entropy = calculate_relative_entropy(main_size, full_blurred_size, saliency_img_size)

        print(f'T (%):{p:.3f} \t S.Ratio: {saliency_img_size:.2f}/{main_size:.2f} \t RE: {saliency_img_size/main_size:.3f} \t Rel.E: {relative_entropy:.3f}')
        normalized_entropy.append(relative_entropy)
    
    normalized_entropy = np.array(normalized_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0, 1)
    percent_entropy = np.stack([pixels, normalized_entropy], axis=0)

    saliency_all_images = np.stack(saliency_all_images, axis=0)

    return saliency_all_images, percent_entropy


if __name__ == "__main__":
    data, labels, categories = load_imgnet_val_data()

    # numpy_image = skimage.io.imread("images/my_junco_scaled.jpg")
    numpy_image = data[100]
    print(numpy_image.shape)
    print(np.min(numpy_image), np.max(numpy_image))

    image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    out_original = io.BytesIO()
    image.save(out_original, format="webp")  # Convert image to webp

    main_size = out_original.tell()

    # Create your mask

    random_img = add_random_pixels(numpy_image, p=0.01)
    print(random_img.shape)
    mask_rand = (ma.array(random_img[:, :, 2]) == 0).data
    print("Mask Shape:", mask_rand.shape)

    interp_mode =  'linear'  # 'nearest'
    random_img_copy = random_img.copy()
    random_img_copy[:, :, 0] = interpolate_missing_pixels(random_img_copy[:, :, 0], mask_rand, method=interp_mode)
    random_img_copy[:, :, 1] = interpolate_missing_pixels(random_img_copy[:, :, 1], mask_rand, method=interp_mode)
    random_img_copy[:, :, 2] = interpolate_missing_pixels(random_img_copy[:, :, 2], mask_rand, method=interp_mode)

    img_pairs, percent_entropy = generate_saliency_focused_images(numpy_image, random_img, mask_rand, \
                                main_size, p_start = 0.0, p_end = 1.0, interp_mode=interp_mode)

    np.save('saliency_focused_images.npy', img_pairs)
    np.save("percent_vs_entropy.npy", percent_entropy)

    # img_pairs=np.load('saliency_focused_images.npy')
    # percent_entropy = np.load("percent_vs_entropy.npy")

    fig, ax = plt.subplots(2, 10, figsize=(9,2))

    m, n = img_pairs.shape[0], img_pairs.shape[1]

    for i in range(10):
        for j in range(2):
            image = img_pairs[i, j, :, :]
            # print(image.shape)
            ax[j, i].imshow(image, interpolation=None, aspect='equal', cmap=plt.cm.inferno)
            ax[j, i].axes.xaxis.set_ticks([])
            ax[j, i].axes.yaxis.set_ticks([])

    plt.show()

    fig = plt.figure()
    plt.plot(percent_entropy[0], percent_entropy[1])
    ax = plt.gca()
    ax.set_xlabel('Threshold %')
    ax.set_title('Threshold vs Entropy')
    ax.set_yticks(np.linspace(0.4, 1.0, num=7))
    ax.set_yticks(np.linspace(0.4, 1.0, num=7))
    ax.set_ylabel('Relative Entropy')

    plt.show()