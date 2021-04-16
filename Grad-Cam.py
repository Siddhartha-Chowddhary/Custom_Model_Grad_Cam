import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt 
from skimage.transform import resize 

# Display
from IPython.display import Image, display
import matplotlib.cm as cm
from vis.utils import utils
from vis.visualization import visualize_cam

import matplotlib.pyplot as plt
# from vis.utils import utils 
# from vis.visualization import visualize_cam
tf.compat.v1.disable_eager_execution()

img_path = 'E:/Project/V1/Data/DATA/Test_Data/car/car.jpg'

img_size = (500, 500)

model = load_model('./Models/Object_classifier.h5')
# last_conv_layer_name = model.layers[12]
last_conv_layer_name = "dense_1"
print(last_conv_layer_name)
model.summary()
# display(Image(img_path))


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size, color_mode = "grayscale")
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)

    print("Array Shape:", array.shape)

    return array

classes = ['car', 'bird','plane', 'trees', 'flowers', 
           'flags', 'Nodule', 'painting', 'numberplate', 'rooms']

img = keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode = "grayscale")
img_array = keras.preprocessing.image.img_to_array(img)

IMG  = plt.imread(img_path) 
Resized_Image = resize(IMG, (500, 500, 1))
pred = model.predict(np.array([Resized_Image,]))
print(model.inputs)
print(model.get_layer(last_conv_layer_name).output)
print(model.output)
#         pred = np.argmax(pred)
index = np.argsort(pred[0,:])
predicted_label = classes[index[9]]
prediction_accuracy = f'{pred[0, index[9]]*100} %' 
print(predicted_label, ' ', prediction_accuracy)

# Remove last layer's softmax

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_1')
# Swap softmax with linear
model.layers[layer_idx].activation = keras.activations.relu
model = utils.apply_modifications(model)


penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_5") 
class_idx  = index
seed_input = img_array
grad_top1  = visualize_cam(model, layer_idx, class_idx, seed_input, 
                           penultimate_layer_idx = penultimate_layer_idx,#None,
                           backprop_modifier     = None,
                           grad_modifier         = None)

def plot_map(grads):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.35)
    fig.colorbar(i)
    print(axes[0].imshow(img))
    plt.show()
plot_map(grad_top1)
