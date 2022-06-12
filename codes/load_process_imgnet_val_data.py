import pickle
import matplotlib.pyplot as plt
import skimage.filters
import numpy as np
import cv2
import tqdm
import os
import joblib

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

# For processing the compressed val image files

# with open('../imgnet_val_data/val224_compressed.pkl', 'rb') as f:
#     d = pickle.load(f)
#
#
# data224 = []
# for img, target in tqdm.tqdm(zip(d['data'], d['target']), total=50000):
#     img224 = str2img(img)
#     data224.append(img224)
#
# data_dict224 = dict(
#     data = np.array(data224).transpose(0, 3, 1, 2),
#     target = d['target']
# )
# joblib.dump(data_dict224, os.path.join('../imgnet_val_data/', 'val224.pkl'))

# close the file
# f.close()

def load_imgnet_val_data():

    with open('./imgnet_val_data/val224.pkl', 'rb') as f:
        d = joblib.load(f)

    # Read the imageNet categories
    with open("./imgnet_val_data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    data = d['data'].transpose(0, 2, 3, 1)
    labels = d['target']
    print(data.shape)
    f.close()

    return data, labels, categories


if __name__ == "__main__":

    # Load the data
    data, labels, categories = load_imgnet_val_data()
    # display the image
    fig = plt.figure(figsize=(12, 12))

    start = 800
    # We plot 20 images from our train_dataset
    for idx in np.arange(start, start+100):
        ax = fig.add_subplot(10, 10, (idx+1) - start, xticks=[], yticks=[])
        plt.imshow(data[idx]) #converting to numpy array as plt needs it.
        ax.set_title(categories[labels[idx].item()], size=6)

    plt.show()