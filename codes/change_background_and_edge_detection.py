from PIL import Image
from torchvision import transforms
import torch
import tensorflow
import pixellib
from pixellib.tune_bg import alter_bg
import cv2
import numpy as np

from cv2 import GaussianBlur

import os

preprocess_edge_detection = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# indices = [2, 4, 19, 20, 27, 40, 55, 64, 83, 94, 105, 145, 169, 172, 242, 250, 256, 259, 260, 263]

# indices_bg_expr = [2, 19, 20, 40,  55, 64, 83, 105, 250, 256]

    
def compute_sobel_edges(images):
    # Read the original image

    all_sobel_edges = []

    for (idx, img) in enumerate(images):
        
        if isinstance(img, np.ndarray):
            pass
        else:
            img = np.asarray(img)
            
        print(np.max(img))

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
        print(img_blur.shape)

        # Sobel Edge Detection

        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
        print(type(sobelxy))

        all_sobel_edges.append(sobelxy)

    all_sobel_edges = np.stack(all_sobel_edges, axis=0)
    print(all_sobel_edges.shape)
    
    return all_sobel_edges


def detect_sobel_edges(basename):
    # Read the original image

    all_sobel_edges = []

    for idx in range(len(indices_bg_expr)):
        f_name = basename + '00000' + str(indices_bg_expr[idx]).zfill(3)+ ".JPEG"
        print(f_name)
        img = Image.open(f_name)
        img = preprocess_edge_detection(img)

        img = np.asarray(img)
        print(np.max(img))

        # img = cv2.imread(f_name)

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
        print(img_blur.shape)

        # Sobel Edge Detection

        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
        print(type(sobelxy))

        all_sobel_edges.append(sobelxy)

        # img_path_list = f_name.split('test/ILSVRC2012_test')  # for Sobel on main images

        img_path_list = f_name.split('cbg_cb')
        print("Image Path Elements: ", img_path_list)

        # save_path = img_path_list[0] + "bg_changed_n_resized/ILSVRC2012_sobel_edge" + img_path_list[1]  # for main images

        save_path = img_path_list[0] + "cbg_cb_sobel_edge" + img_path_list[1]
        print("Saving here: %s" % save_path)
        cv2.imwrite(save_path, sobelxy)

    all_sobel_edges = np.stack(all_sobel_edges, axis=0)
    print(all_sobel_edges.shape)
    np.save("/Users/mdmahfuzurrahman/Downloads/bg_changed_n_resized/method_research_imgNet_cbg_cb_sobel_edges.npy", all_sobel_edges)



def resize_bg(path):
    bg_image = Image.open(path)
    bg_tensor = preprocess(bg_image)

    sample_img = transforms.ToPILImage()(bg_tensor.squeeze_(0))

    full_path = path[:-5] + "_resized" + ".jpg"
    print('Full Path: %s' % full_path)
    sample_img.save(full_path)

def image_resize(basename):

    """
    resize image and background to be on the same size before background replacement
    :return:
    :rtype:
    """

    for idx in range(len(indices)):
        f_name = basename + '00000' + str(indices[idx]).zfill(3)+ ".JPEG"
        print(f_name)
        input_image = Image.open(f_name)
        input_tensor = preprocess(input_image)


        filename_list = f_name.split('test')
        sample_img = transforms.ToPILImage()(input_tensor.squeeze_(0))
        path = filename_list[0]+"bg_changed_n_resized"+"/"
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("path just created!!")
        full_path = filename_list[0]+"bg_changed_n_resized"+filename_list[1]+"resized_"+filename_list[2][1:]
        print('Full Path: %s' %full_path)
        sample_img.save(full_path)


def change_background(basename):
    for idx in range(len(indices)):
        img_path = basename + '00000' + str(indices[idx]).zfill(3) + ".JPEG"
        print(img_path)

        img_path_list = img_path.split('ILSVRC2012_resized')
        print("Image Path Elements: ", img_path_list)
        change_bg = alter_bg()
        change_bg.load_pascalvoc_model("/Users/mdmahfuzurrahman/Downloads/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        output = change_bg.change_bg_img(f_image_path = img_path, b_image_path = "/Users/mdmahfuzurrahman/Downloads/bg_changed_n_resized/bg_cb_resized.jpg")
        type(output)
        save_path = img_path_list[0]+"ILSVRC2012_cbg_cb"+img_path_list[1]
        print("Saving here: %s" % save_path)
        cv2.imwrite(save_path, output)


# resize_bg('/Users/mdmahfuzurrahman/Downloads/bg_cb.jpeg')
# basename_main = '/Users/mdmahfuzurrahman/Downloads/test/ILSVRC2012_test_'
# basename_bgc_images = "/Users/mdmahfuzurrahman/Downloads/bg_changed_n_resized/ILSVRC2012_cbg_cb_"


# image_resize(basename_main)
# basename_bgc = '/Users/mdmahfuzurrahman/Downloads/bg_changed_n_resized/ILSVRC2012_resized_'

# change_background(basename_bgc)

# detect_sobel_edges(basename_bgc_images)




