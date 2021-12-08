
# coding: utf-8

# In[25]:

import cv2
import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

# import sys
import os
# dir = '/Users/LSK/PycharmProjects/panet/'
# sys.path.append(dir)

from ssd import SSD300
from ssd_utils import BBoxUtility

#get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.45
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# In[9]:

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
print(NUM_CLASSES)


# In[26]:

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(os.path.join(dir,'weights_SSD300.hdf5'), by_name=True)

# print(model.get_layer('mbox_conf').output_shape)

bbox_util = BBoxUtility(NUM_CLASSES)


# In[27]:

inputs = []
images = []
img_path = os.path.join(dir,'./pics/fish-bike.jpg')
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img.copy())
img_path = os.path.join(dir,'./pics/cat.jpg')
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img.copy())
img_path = os.path.join(dir,'./pics/boys.jpg')
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img.copy())
img_path = os.path.join(dir,'./pics/car_cat.jpg')
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img.copy())
img_path = os.path.join(dir,'./pics/car_cat2.jpg')
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))
# print(inputs[0].shape)


# In[28]:

preds = model.predict(inputs, batch_size=1, verbose=1)


# In[29]:

results = bbox_util.detection_out(preds)


# In[30]:

# %%time
# a = model.predict(inputs, batch_size=1)
# b = bbox_util.detection_out(preds)


# In[32]:

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [j for j, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()


# In[ ]:



