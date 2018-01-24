from vis.visualization import visualize_saliency, visualize_cam, visualize_activation
from scipy.misc import imread, imsave, imresize
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os


def boxing(img, label):
    model_path = "./models/resnet50_model.h5"
    model = load_model(model_path)
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_2"][0]

    heatmap = visualize_saliency(model, layer_idx, np.expand_dims(label, axis=0), img)
    # heatmap = visualize_activation(model, layer_idx, np.expand_dims(2, axis=0), img)
    # heatmap = visualize_cam(model, layer_idx, np.expand_dims(230, axis=0), img)

    # plt.imshow(heatmap, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()
    # fig.savefig( os.path.join("saliency_map/", IMG_ID +".png"), dpi=100)

    k_size = 28
    k = np.ones((k_size,k_size))/k_size
    heatmap = signal.convolve2d(heatmap[:,:,0], k, boundary='wrap', mode='same')/k.sum()

    # plt.imshow(heatmap, cmap=plt.cm.jet)
    # plt.show()

    threshold = heatmap.max()*0.3

    maxTop = maxLeft = 999999999
    maxRight = maxBottom = -1
    for h in range(224):
        for w in range(224):
            # print(h,w)
            if heatmap[h][w] > threshold:
                if h < maxTop: maxTop = h
                if h > maxBottom: maxBottom = h
                if w < maxLeft: maxLeft = w
                if w > maxRight: maxRight = w


    maxTop = int(maxTop/3)
    maxBottom = int(maxBottom/3)
    maxLeft = int(maxLeft/3)
    maxRight = int(maxRight/3)

    for h in range(224):
        for w in range(224):
            if (int(h/3) == maxTop and int(w/3) in range(maxLeft, maxRight)) or (int(h/3) == maxBottom and int(w/3) in range(maxLeft, maxRight)) or (int(w/3) == maxRight and int(h/3) in range(maxTop, maxBottom))  or (int(w/3) == maxLeft and int(h/3) in range(maxTop, maxBottom)):
                img[h][w][0] = img[h][w][1] = 255
                img[h][w][2] = 0

    return img


# sample usage
data_path = 'data/train' 
img = imread( os.path.join(data_path, 'Helianthus annuus 向日葵/C1326_30.jpg'), mode ='RGB')
img = imresize(img ,size=(224,224))
result = boxing(img, 107)


plt.imshow(result, cmap=plt.cm.jet)
plt.show()