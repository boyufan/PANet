from panet import PANET
from ssd.ssd import SSD300
from keras.utils.visualize_util import plot
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import keras

import cv2
import pickle
from matplotlib.pyplot import imread

from ssd.ssd_training import MultiboxLoss

PVEC = np.array([0.833, 0.346, 0.189, 0.098, 0.934, 0.679, 
                         0.481, 0.875, 0.081, 0.579, 0.901, 0.223])

SCAT_COCO = ['outdoor', 'food', 'indoor', 'appliance', 'sports', 'person', 
                  'animal', 'vehicle', 'furniture', 'accessory', 'electronic', 'kitchen']

CAT_COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
            'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
            'donut', 'cake', 'chair', 'couch', 'potted plant', 
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(CAT_COCO) + 1

cat2scat = pickle.load(open('pickle_files/cat2scat.pickle', 'rb'))
## cat, scat pair: (CAT_COCO[i], SCAT_COCO[cat2scat[CAT_COCO[i]]])
## preference vector for detailed class
# pvec_cat = np.zeros((NUM_CLASSES-1,1))
# for i in range(NUM_CLASSES-1):
#     pvec_cat[i] = PVEC[cat2scat[CAT_COCO[i]]]
print('Building scat2cat mapping...')
## id mapping
c2sc = {}
for i in range(NUM_CLASSES-1):
      c2sc[i] = cat2scat[CAT_COCO[i]]
scat2cat = {}
for k,v in c2sc.items():
      scat2cat[v] = scat2cat.get(v, [])
      scat2cat[v].append(k)

# inputs = []
# images = []
# img_path = '/Users/LSK/Downloads/book_covers/RE/001.jpg'
# img = image.load_img(img_path, target_size=(300, 300))
# img = image.img_to_array(img)
# images.append(imread(img_path))
# inputs.append(img.copy())
# inputs = preprocess_input(np.array(inputs))

input_shape = (300, 300, 3)

# load weights for training
base_model = SSD300(input_shape, num_classes=NUM_CLASSES)
base_model.load_weights('weights_SSD300.hdf5', by_name=True)
base_layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
# load saliency prior
prior_sal = np.load('prior_sal.npy')

model = PANET(input_shape, prior_sal, scat2cat, 
                  PVEC, num_classes=NUM_CLASSES)
# print(model.summary())
layer_dict = dict([(layer.name, layer) for layer in model.layers])
# set weigths
for layer_name in base_layer_dict.keys():
      # print("current layer name", layer_name)
      if layer_name != 'input_1' and layer_name != 'zeropadding2d_1':
          layer_dict[layer_name].set_weights(base_layer_dict[layer_name].get_weights())
          # freeze weigths
          layer_dict[layer_name].trainable = False

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

model.fit_generator()