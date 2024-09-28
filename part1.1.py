#import scipy as sp

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import io, color
import cv2
from align_image_code import align_images



# helper function for displaying n images with n titles and color maps on a plot
def display_n_images(images: list, titles: list[str], cmaps = []) -> bool:
    width = len(images)
    cmap_len = len(cmaps)
    if width != len(titles) or (cmap_len != 0 and cmap_len != width):
        return False
    plt.figure(figsize=(4 * width, 6))
    for i in range(1, width+1):
        plt.subplot(1, width, i)
        plt.title(titles[i - 1])
        if cmap_len == 0 or cmaps[i - 1] == 'default' or cmaps[i - 1] == '':
            plt.imshow(images[i - 1])
        else:
            plt.imshow(images[i - 1], cmap=cmaps[i - 1])
    plt.show()
    return True

# PART 1
# Load the image and convert to grayscale
gray_image = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)

# Define finite difference operators
D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])

# Convolve the image with the operators
partial_x = convolve2d(gray_image, D_x, mode='same', boundary='symm')
partial_y = convolve2d(gray_image, D_y, mode='same', boundary='symm')

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(partial_x**2 + partial_y**2)

# Define a threshold
threshold = 12
edge_image = gradient_magnitude > threshold

plt.figure(figsize=(20, 6))
plt.subplot(1, 5, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.subplot(1, 5, 2)
plt.title('Partial Derivative X')
plt.imshow(partial_x, cmap='gray')
plt.subplot(1, 5, 3)
plt.title('Partial Derivative Y')
plt.imshow(partial_y, cmap='gray')
plt.subplot(1, 5, 4)
plt.title('Gradient Magnitude Image')
plt.imshow(gradient_magnitude, cmap='gray')
plt.subplot(1, 5, 5)
plt.title('Edge Image (Threshold = 12)')
plt.imshow(edge_image, cmap='gray')
plt.show()

plt.figure(figsize=(20, 6))
plt.subplot(1, 5, 1)
plt.title('Threshold=8')
plt.imshow(gradient_magnitude > 8, cmap='gray')
plt.subplot(1, 5, 2)
plt.title('Threshold=10')
plt.imshow(gradient_magnitude > 10, cmap='gray')
plt.subplot(1, 5, 3)
plt.title('Threshold=12')
plt.imshow(gradient_magnitude > 12, cmap='gray')
plt.subplot(1, 5, 4)
plt.title('Threshold=15')
plt.imshow(gradient_magnitude > 15, cmap='gray')
plt.subplot(1, 5, 5)
plt.title('Threshold=25')
plt.imshow(gradient_magnitude > 25, cmap='gray')
plt.show()

# creates a gaussian filter from a kernel and use it to blur the cameraman image
# this will be used as the starting image for the next two transformations
g_kernel = cv2.getGaussianKernel(9, 1.5)
g_filter = np.outer(g_kernel, np.transpose(g_kernel))
g_blur_image = convolve2d(gray_image, g_filter, mode='same', boundary='symm')

# derives the gradient magnitude and edge image using partial derivatives, like earlier
g_partial_x = convolve2d(g_blur_image, D_x, mode='same', boundary='symm')
g_partial_y = convolve2d(g_blur_image, D_y, mode='same', boundary='symm')
g_blur_magnitude = np.sqrt(g_partial_x**2 + g_partial_y**2)
g_edge_image = g_blur_magnitude > threshold

# derives the gradient magnitude and edge image by first taking a derivative
# of the Gaussian filter with D_x and D_y, with results being nearly identical
dog_x = convolve2d(g_filter, D_x, mode='same', boundary='symm')
dog_y = convolve2d(g_filter, D_y, mode='same', boundary='symm')
gradient_magnitude_dog = np.sqrt(convolve2d(g_blur_image, dog_x, mode='same', boundary='symm')**2 + convolve2d(g_blur_image, dog_y, mode='same', boundary='symm')**2)
edge_image_dog = gradient_magnitude_dog > threshold

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Gaussian Filter (9x9)')
plt.imshow(g_filter, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Gaussian Blur (Sigma = 1.5)')
plt.imshow(g_blur_image, cmap='gray')
plt.show()

plt.figure(figsize=(20, 6))
plt.subplot(1, 5, 1)
plt.title('Gaussian Blur')
plt.imshow(g_blur_image, cmap='gray')
plt.subplot(1, 5, 2)
plt.title('Partial Derivative X')
plt.imshow(g_partial_x, cmap='gray')
plt.subplot(1, 5, 3)
plt.title('Partial Derivative Y')
plt.imshow(g_partial_y, cmap='gray')
plt.subplot(1, 5, 4)
plt.title('Gradient Magnitude Image')
plt.imshow(g_blur_magnitude, cmap='gray')
plt.subplot(1, 5, 5)
plt.title('Edge Image (Threshold = 12)')
plt.imshow(g_edge_image, cmap='gray')
plt.show()

plt.figure(figsize=(20, 6))
plt.subplot(1, 5, 1)
plt.title('Gaussian Blur')
plt.imshow(g_blur_image, cmap='gray')
plt.subplot(1, 5, 2)
plt.title('Deriv. of Gauss. X')
plt.imshow(dog_x, cmap='gray')
plt.subplot(1, 5, 3)
plt.title('Deriv. of Gauss. Y')
plt.imshow(dog_y, cmap='gray')
plt.subplot(1, 5, 4)
plt.title('Gradient Magnitude Image')
plt.imshow(gradient_magnitude_dog, cmap='gray')
plt.subplot(1, 5, 5)
plt.title('Edge Image (Threshold = 12)')
plt.imshow(edge_image_dog, cmap='gray')
plt.show()


# PART 2
def sharpen(filename, show_steps = False, pre_blur = False, flip = True):
    # reads image and the splits the color channels
    #taj_color = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if type(filename) == str:
        taj_color = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    else:
        taj_color = filename
    if taj_color is None:
        print("Image " + filename + " not found")
        return None
    elif taj_color.shape[2] != 3:
        print("Failed due to incorrect dimensions (make sure file is .jpg): " + str(taj_color.shape))
        return None
    if flip:
        taj_b, taj_g, taj_r = np.split(taj_color, 3, axis=2)
        taj_color = np.dstack([taj_r, taj_g, taj_b])
    else:
        taj_r, taj_g, taj_b = np.split(taj_color, 3, axis=2)
    taj_r = taj_r.reshape(taj_r.shape[0], taj_r.shape[1])
    taj_g = taj_g.reshape(taj_g.shape[0], taj_g.shape[1])
    taj_b = taj_b.reshape(taj_b.shape[0], taj_b.shape[1])
    if pre_blur:
        taj_color_og = taj_color
        g_kernel_preblur = cv2.getGaussianKernel(9, 9)
        g_filter_preblur = np.outer(g_kernel_preblur, np.transpose(g_kernel_preblur))
        g_blur_r = convolve2d(taj_r, g_filter_preblur, mode='same', boundary='symm')
        g_blur_g = convolve2d(taj_g, g_filter_preblur, mode='same', boundary='symm')
        g_blur_b = convolve2d(taj_b, g_filter_preblur, mode='same', boundary='symm')
        taj_color = np.dstack([g_blur_r, g_blur_g, g_blur_b]) / 255
        display_n_images([taj_color_og, taj_color], ['Original', 'Pre-Blurred'])
    if show_steps:
        display_n_images([taj_r, taj_g, taj_b], ['Red Channel', 'Green Channel', 'Blue Channel'], ['gray', 'gray', 'gray'])

    # creates Gaussian kernel, blurs the color channels, and combines for blurred color image
    g_kernel_taj = cv2.getGaussianKernel(9, 2.6)
    g_filter_taj = np.outer(g_kernel_taj, np.transpose(g_kernel_taj))
    unit_impulse = [[0.0 for j in range(9)] for i in range(9)]
    alpha = 2.5
    unit_impulse[4][4] = alpha + 1.0
    l_of_g_taj = (unit_impulse) - (alpha * g_filter_taj)
    g_blur_taj_r = convolve2d(taj_r, g_filter_taj, mode='same', boundary='symm')
    g_blur_taj_g = convolve2d(taj_g, g_filter_taj, mode='same', boundary='symm')
    g_blur_taj_b = convolve2d(taj_b, g_filter_taj, mode='same', boundary='symm')

    g_sharp_taj_r = convolve2d(taj_r, l_of_g_taj, mode='same', boundary='symm')
    g_sharp_taj_g = convolve2d(taj_g, l_of_g_taj, mode='same', boundary='symm')
    g_sharp_taj_b = convolve2d(taj_b, l_of_g_taj, mode='same', boundary='symm')
    g_sharp_image_taj = np.dstack([g_sharp_taj_r, g_sharp_taj_g, g_sharp_taj_b]) / 255
    if show_steps:
        display_n_images([g_blur_taj_r, g_blur_taj_g, g_blur_taj_b], ['Blurred Red Channel', 'Blurred Green Channel', 'Blurred Blue Channel'], ['gray', 'gray', 'gray'])

    # uses blurred and original image to create high pass filtered image
    g_blur_image_taj = np.dstack([g_blur_taj_r, g_blur_taj_g, g_blur_taj_b]) / 255
    if show_steps:
        display_n_images([taj_color, g_filter_taj, g_blur_image_taj], ['Original Image', 'Gaussian Filter', 'Blurred Image'], ['', 'gray', ''])
    hp_image_taj = (taj_color / 255) + 2.5*((taj_color / 255) - g_blur_image_taj)

    # doubles the higher frequencies in proportion to the Gaussian blurred low frequencies
    # the taj_color / 255 parameter is the unit impulse (identity)
    sharpened_taj = cv2.addWeighted(taj_color / 255, 2.0, g_blur_image_taj, -1.0, 0) # (taj_color * 2.0) + (g_blur_image * -1.0) + 0
    if show_steps:
        display_n_images([taj_color, l_of_g_taj, sharpened_taj, hp_image_taj, g_sharp_image_taj], ['Original Image', 'Laplacian of Guassian', 'Weighted Add (Reference)', 'Subtraction Method', 'Single Convolution w/ LoG'], ['', 'gray', '', '', ''])
        if pre_blur:
            display_n_images([taj_color, taj_color_og, g_sharp_image_taj], ['Pre-Blurred Image', 'Original Image', 'Resharpened Image'])
    else:
        if pre_blur:
            display_n_images([taj_color, taj_color_og, g_sharp_image_taj], ['Pre-Blurred Image', 'Original Image', 'Resharpened Image'])
        else:
            display_n_images([taj_color, g_sharp_image_taj], ['Original Image', 'Laplacian of Gaussian Method'])
    return g_sharp_image_taj

sharpen('taj.jpg', show_steps = True)
sharpen('testImg1.jpg')
sharpen('testImg2.jpg')
sharpen('testImg3.jpg')
sharpen('testImg4.jpg', show_steps = False, pre_blur = True)

def filter(img, low_or_high, sigma = 7, g_size = 9):
    r, g, b = np.split(img, 3, axis=2)
    r = r.reshape(r.shape[0], r.shape[1])
    g = g.reshape(g.shape[0], g.shape[1])
    b = b.reshape(b.shape[0], b.shape[1])
    g_kernel = cv2.getGaussianKernel(g_size, sigma)
    g_filter = np.outer(g_kernel, np.transpose(g_kernel))
    blur_r = convolve2d(r, g_filter, mode='same', boundary='symm')
    blur_g = convolve2d(g, g_filter, mode='same', boundary='symm')
    blur_b = convolve2d(b, g_filter, mode='same', boundary='symm')
    blur_img = np.dstack([blur_r, blur_g, blur_b])
    if low_or_high == 'low':
        return blur_img
    elif low_or_high == 'high':
        return img - blur_img
    else:
        return None

def layer(f1, f2, display = False, sigma1 = 18, g_size1 = 36, sigma2 = 18, g_size2 = 36): # f1 = Coarse/Blurry and f2 = Fine/Clear
    if type(f1) == str:
        im1 = plt.imread(f1) / 255.
    else:
        im1 = f1
    if type(f2) == str:
        im2 = plt.imread(f2) / 255
    else:
        im2 = f2
    im1_aligned, im2_aligned = align_images(im2, im1)
    
    #display_n_images([np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1)))), np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2))))], ['im1_log', 'im2_log'], ['gray', 'gray'])
    if display:
        display_n_images([im1_aligned, im2_aligned], ['Aligned Input 1 (Low Freq.)', 'Aligned Input 2 (High Freq.)'])
    #testimg = np.dstack([r, g, b])
    #im1_filter = filter(im1_aligned, 'high')
    #sharp = sharpen(im1_aligned, flip = False)
    #display_n_images([sharp], ['sharpened cat'])
    #r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
    #g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    #b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
    if display:
        r, g, b = np.split(im1, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Input 1', '2D FT Green Channel Input 1', '2D FT Blue Channel Input 1'], ['gray', 'gray', 'gray'])
        r, g, b = np.split(im2, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Input 2', '2D FT Green Channel Input 2', '2D FT Blue Channel Input 2'], ['gray', 'gray', 'gray'])

    im1_filter = filter(im1_aligned, 'high', sigma = sigma2, g_size = g_size2)
    if display:
        display_n_images([im1_aligned, im1_filter], ['Aligned Image', 'High-Pass Filtered Image'])
    if display:
        r, g, b = np.split(im1_filter, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Filtered 1', '2D FT Green Channel Filtered 1', '2D FT Blue Channel Filtered 1'], ['gray', 'gray', 'gray'])

    #r, g, b = np.split(im2_aligned, 3, axis=2)
    #testimg = np.dstack([r, g, b])
    im2_filter = filter(im2_aligned, 'low', sigma = sigma1, g_size = g_size1)
    if display:
        display_n_images([im2_aligned, im2_filter], ['Aligned Image', 'Low-Pass Filtered Image'])
    if display:
        r, g, b = np.split(im2_filter, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Filtered 2', '2D FT Green Channel Filtered 2', '2D FT Blue Channel Filtered 2'], ['gray', 'gray', 'gray'])
    #display_n_images([im2_aligned, testimg, im2_filter], ['', '', ''])
    add_img = im1_filter + im2_filter
    avg_img = (im1_filter + im2_filter)/2
    if display:
        r, g, b = np.split(avg_img, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Average Method', '2D FT Green Channel Average Method', '2D FT Blue Channel Average Method'], ['gray', 'gray', 'gray'])
        r, g, b = np.split(add_img, 3, axis=2)
        r_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(r))))
        g_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))))
        b_in_log = np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))))
        display_n_images([r_in_log, g_in_log, b_in_log], ['2D FT Red Channel Add Method', '2D FT Green Channel Add Method', '2D FT Blue Channel Add Method'], ['gray', 'gray', 'gray'])
    display_n_images([im1_aligned, im2_aligned, (im1_filter + im2_filter)/2, im1_filter + im2_filter], ['Low Freq. Image', 'High Freq. Image', 'layered average', 'layered add'])

layer('DerekPicture.jpg', 'nutmeg.jpg')
layer('img2.jpg', 'img1.jpg', display = True) # change back to true
layer('img3_2.jpg', 'img4.jpg', sigma2 = 9, g_size2 = 9)
layer('img4.jpg', 'img3_2.jpg', sigma2 = 9, g_size2 = 9)
layer('transitionpart1.jpg', 'transitionpart2.jpg')
layer('transitionpart2.jpg', 'transitionpart1.jpg')

def g_stack(img, n, g_size = 9, sigma = 3, flip = False, resize = False):
    if flip:
        b, g, r = np.split(img, 3, axis = 2)
        img = np.dstack([r, g, b])
    result = [img]
    for i in range(1, n):
        g_kernel = cv2.getGaussianKernel(g_size, sigma)
        g_filter = np.outer(g_kernel, np.transpose(g_kernel))
        r, g, b = np.split(result[-1], 3, axis = 2)
        r = r.reshape(r.shape[0], r.shape[1])
        g = g.reshape(g.shape[0], g.shape[1])
        b = b.reshape(b.shape[0], b.shape[1])
        blur_r = convolve2d(r, g_filter, mode='same', boundary='symm')
        blur_g = convolve2d(g, g_filter, mode='same', boundary='symm')
        blur_b = convolve2d(b, g_filter, mode='same', boundary='symm')
        img2 = np.dstack([blur_r, blur_g, blur_b])
        if resize:
            g_size = (g_size * 2) - 1
            sigma *= 2
        result.append(img2)
    return result

def normalize_image(img):
    r, g, b = np.split(img, 3, axis = 2)
    r = r.reshape(r.shape[0], r.shape[1])
    g = g.reshape(g.shape[0], g.shape[1])
    b = b.reshape(b.shape[0], b.shape[1])
    #d2arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    r_min = np.min(r, axis = (0, 1), keepdims=True)
    r_max = np.max(r, axis = (0, 1), keepdims=True)
    g_min = np.min(g, axis = (0, 1), keepdims=True)
    g_max = np.max(g, axis = (0, 1), keepdims=True)
    b_min = np.min(b, axis = (0, 1), keepdims=True)
    b_max = np.max(b, axis = (0, 1), keepdims=True)
    #print(np.min(d2arr, axis = (0, 1), keepdims=True))
    r = (r - r_min) / (r_max - r_min)
    g = (g - g_min) / (g_max - g_min)
    b = (b - b_min) / (b_max - b_min)
    #r = r + (-1 * r_min)
    #g = g + (-1 * g_min)
    #b = b + (-1 * b_min)
    return np.dstack([r, g, b])

def l_stack(g_stack, fixed_add = 0, mask = None, g_size = 19, sigma = 18, resize = False, enhance = False):
    result = []
    g_kernel = cv2.getGaussianKernel(g_size, sigma)
    g_filter = np.outer(g_kernel, np.transpose(g_kernel))
    temp_mask = convolve2d(mask, g_filter, mode='same', boundary='symm')
    for i in range(len(g_stack) - 1):
        #result.append(cv2.subtract(g_stack[i], g_stack[i + 1]) + 0.4)
        #result.append(normalize_image(cv2.subtract(g_stack[i], g_stack[i + 1])))
        if enhance == False:
            res_img = cv2.subtract(g_stack[i], g_stack[i + 1])
        else:
            if fixed_add > 0:
                res_img = cv2.subtract(g_stack[i], g_stack[i + 1]) + fixed_add
                #result.append(cv2.subtract(g_stack[i], g_stack[i + 1]) + fixed_add) # prev code
            else:
                res_img = normalize_image(cv2.subtract(g_stack[i], g_stack[i + 1]))
                #result.append(normalize_image(cv2.subtract(g_stack[i], g_stack[i + 1]))) # prev code
        if g_size != 0 and sigma != 0 and enhance == False and mask is not None:
            #print("entered mask option, i: " + str(i))
            if resize:
                g_kernel = cv2.getGaussianKernel(g_size, sigma)
                g_filter = np.outer(g_kernel, np.transpose(g_kernel))
                temp_mask = convolve2d(mask, g_filter, mode='same', boundary='symm')
            print("convolution: " + str(i))
            r, g, b = np.split(res_img, 3, axis = 2)
            r = r.reshape(r.shape[0], r.shape[1])
            g = g.reshape(g.shape[0], g.shape[1])
            b = b.reshape(b.shape[0], b.shape[1])
            #r *= convolve2d(r, temp_mask, mode='same', boundary='symm')
            #print("convolution 2, i: " + str(i))
            #g *= convolve2d(g, temp_mask, mode='same', boundary='symm')
            #print("convolution 3, i: " + str(i))
            #b *= convolve2d(b, temp_mask, mode='same', boundary='symm')
            #print("convolution 4, i: " + str(i))
            r *= temp_mask
            g *= temp_mask
            b *= temp_mask
            res_img = np.dstack([r, g, b])
            if resize:
                g_size = (g_size * 2) - 1
                sigma *= 2
        #sub = cv2.subtract(g_stack[i], g_stack[i + 1])
        #print("shape: " + str(norm.shape))
        #r, g, b = np.split(sub, 3, axis = 2)
        result.append(res_img)
    tmp_img = g_stack[-1]
    if g_size != 0 and sigma != 0:
        #tmp_img = g_stack[-1]
        r, g, b = np.split(tmp_img, 3, axis = 2)
        r = r.reshape(r.shape[0], r.shape[1])
        g = g.reshape(g.shape[0], g.shape[1])
        b = b.reshape(b.shape[0], b.shape[1])
        r *= temp_mask
        g *= temp_mask
        b *= temp_mask
        tmp_img = np.dstack([r, g, b])
    #result.append(g_stack[-1])
    result.append(tmp_img)
    return result

def blend(f1, f2, flip = False, show_steps = False, mask1 = None, mask2 = None, g_size = 81, sigma = 63, resize = True, enhance = False, stack_size = 5, vert_sym = False, fixed_add = 0.0):
    if type(f1) == str:
        left_img = plt.imread(f1) / 255.
    else:
        left_img = f1
    if type(f2) == str:
        right_img = plt.imread(f2) / 255
    else:
        right_img = f2
    if left_img.shape != right_img.shape:
        left_img, right_img = align_images(left_img, right_img)
    left_g_stack = g_stack(left_img, stack_size, flip = flip, resize = resize)
    right_g_stack = g_stack(right_img, stack_size, flip = flip, resize = resize)
    if show_steps:
        display_n_images(left_g_stack, ['Gaussian Stack for First Input Image', '', '', '', ''])
        display_n_images(right_g_stack, ['Gaussian Stack for Second Input Image', '', '', '', ''])
    if vert_sym:
        left_mask = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.float32)
        left_mask[:, :(left_img.shape[1] // 2)] = 1
        right_mask = np.zeros((right_img.shape[0], right_img.shape[1]), dtype=np.float32)
        right_mask[:, (right_img.shape[1] // 2):] = 1
    if mask1 is not None and mask2 is not None:
        left_mask = mask1
        right_mask = mask2
    l_stack1 = l_stack(left_g_stack, fixed_add = fixed_add, mask = left_mask, g_size = g_size, sigma = sigma, resize = not resize)
    l_stack2 = l_stack(right_g_stack, fixed_add = fixed_add, mask = right_mask, g_size = g_size, sigma = sigma, resize = not resize)
    l_stack1_sum = sum(l_stack1)
    l_stack2_sum = sum(l_stack2)
    result_image = l_stack1_sum + l_stack2_sum
    if show_steps:
        #display_n_images()
        #display_n_images()
        #display_n_images()
        #plt.figure(figsize=(12, 30))
        for i in range(len(l_stack1)):
            plt.subplot(stack_size + 1, 3, (i * 3) + 1)
            plt.title('')
            if i == len(l_stack1) - 1:
                plt.imshow(l_stack1[i])
            else:
                plt.imshow(l_stack1[i] + 0.4)
            plt.subplot(stack_size + 1, 3, (i * 3) + 2)
            plt.title('')
            if i == len(l_stack1) - 1:
                plt.imshow(l_stack2[i])
            else:
                plt.imshow(l_stack2[i] + 0.4)
            plt.subplot(stack_size + 1, 3, (i * 3) + 3)
            plt.title('')
            if i == len(l_stack1) - 1:
                plt.imshow(l_stack1[i] + l_stack2[i])
            else:
                plt.imshow(l_stack1[i] + l_stack2[i] + 0.4)
            #display_n_images([l_stack1[i] + 0.4, l_stack2[i] + 0.4, l_stack1[i] + l_stack2[i]], ['First Component', 'Second Component', 'Sum'])
        plt.subplot(stack_size + 1, 3, (stack_size * 3) + 1)
        plt.title('')
        plt.imshow(l_stack1_sum)
        plt.subplot(stack_size + 1, 3, (stack_size * 3) + 2)
        plt.title('')
        plt.imshow(l_stack2_sum)
        plt.subplot(stack_size + 1, 3, (stack_size * 3) + 3)
        plt.title('')
        plt.imshow(result_image)
        plt.show()
    display_n_images([result_image], ['Blended Image'])

def read(filename, flip = False, gray = False, norm = True):
    if gray:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else: 
        image = cv2.imread(filename)
    if norm:
        image = image / 255
    if flip and not gray:
        b, g, r = np.split(image, 3, axis=2)
        r = r.reshape(r.shape[0], r.shape[1])
        g = g.reshape(g.shape[0], g.shape[1])
        b = b.reshape(b.shape[0], b.shape[1])
        image = np.dstack([r, g, b])
    return image

blend('apple.jpeg', 'orange.jpeg', flip = False, show_steps = True, enhance = False, stack_size = 5, vert_sym = True)
blend('img4.jpg', 'img3_2.jpg', flip = False, show_steps = False, enhance = False, stack_size = 5, vert_sym = True)
blend('nutmeg.jpg', 'DerekPicture.jpg', flip = False, show_steps = False, enhance = False, stack_size = 5, vert_sym = True)
orange = read('nutmeg.jpg', flip = True)
derek = read('DerekPicture.jpg', flip = True)
#display_n_images([orange, derek], ['', ''])
aligned_orange, aligned_derek = align_images(orange, derek)
#display_n_images([aligned_orange, aligned_derek], ['', ''])
irr_mask_derek = np.zeros((aligned_derek.shape[0], aligned_derek.shape[1]), dtype=np.float32)
plt.imshow(aligned_derek)
cc = plt.ginput(1)
print(cc)
plt.close()
cc0, cc1 = int(cc[0][0]), int(cc[0][1])
print(int(0.6 * min(aligned_derek.shape[0], aligned_derek.shape[1])))
r_ratio = 0.45
irr_mask_derek = cv2.circle(irr_mask_derek, (cc0, cc1), int(r_ratio * min(aligned_derek.shape[0], aligned_derek.shape[1])), 1, int(r_ratio * min(aligned_derek.shape[0], aligned_derek.shape[1])))
print(irr_mask_derek)
irr_mask_orange = 1 - irr_mask_derek
print(irr_mask_orange)
display_n_images([irr_mask_derek, irr_mask_orange], ['', ''], ['gray', 'gray'])
blend(aligned_orange, aligned_derek, flip = False, show_steps = True, mask1 = irr_mask_derek, mask2 = irr_mask_orange, enhance = False, stack_size = 5)

#irregular_mask = np.zeros()
"""
images = g_stack(cv2.imread('apple.jpeg') / 255., 5, flip = True, resize = True, g_size = 9, sigma = 8)
images2 = g_stack(cv2.imread('orange.jpeg') / 255., 5, flip = True, resize = False, g_size = 19, sigma = 18)
images3 = g_stack(cv2.imread('orange.jpeg') / 255., 5, flip = True, resize = True, g_size = 9, sigma = 8)
images4 = g_stack(cv2.imread('apple.jpeg') / 255., 5, flip = True, resize = False, g_size = 19, sigma = 18)
display_n_images(l_stack(images, fixed_add = 0.4), ['Fixed-add Resizing-K Laplacian' for x in range(5)])
display_n_images(l_stack(images2, fixed_add = 0.4), ['Fixed-add Fixed-K Laplacian' for x in range(5)])
display_n_images(l_stack(images), ['Fixed-add Resizing-K Laplacian' for x in range(5)])
display_n_images(l_stack(images2), ['Fixed-add Fixed-K Laplacian' for x in range(5)])
display_n_images(l_stack(images3, fixed_add = 0.4), ['Fixed-add Resizing-K Laplacian' for x in range(5)])
display_n_images(l_stack(images4, fixed_add = 0.4), ['Fixed-add Fixed-K Laplacian' for x in range(5)])
display_n_images(l_stack(images3), ['Fixed-add Resizing-K Laplacian' for x in range(5)])
display_n_images(l_stack(images4), ['Fixed-add Fixed-K Laplacian' for x in range(5)])
testimg = cv2.imread('apple.jpeg')
testimg2 = cv2.imread('orange.jpeg')
seam_mask = np.zeros((testimg.shape[0], testimg.shape[1]), dtype=np.float32)
seam_mask[:, :testimg.shape[1] // 2] = 1
seam_mask2 = np.zeros((testimg2.shape[0], testimg2.shape[1]), dtype=np.float32)
seam_mask2[:, (testimg2.shape[1] // 2):] = 1
result_image = sum(l_stack(images, fixed_add = 0.0, mask = seam_mask, g_size = 81, sigma = 63, resize = False)) + sum(l_stack(images3, fixed_add = 0.0, mask = seam_mask2, g_size = 81, sigma = 63, resize = False))
l_stack(images3, fixed_add = 0.0, mask = seam_mask2, g_size = 81, sigma = 63, resize = False)
display_n_images([result_image], [''])
display_n_images(l_stack(images, fixed_add = 0.0, mask = seam_mask, g_size = 81, sigma = 63, resize = False), ['Testing Mask Laplacian' for x in range(5)])
"""
