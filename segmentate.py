# Copied from https://nnabla.readthedocs.io/en/latest/python/api/models/semantic_segmentation.html


#Import reauired modules
import numpy as np
import nnabla as nn
from nnabla.utils.image_utils import imread
from nnabla.models.semantic_segmentation import DeepLabV3plus
from nnabla.models.semantic_segmentation.utils import ProcessImage

target_h = 513
target_w = 513
# Get context
from nnabla.ext_utils import get_extension_context
# comment out the code below which enables GPU, because having no GPU.
# nn.set_default_context(get_extension_context('cudnn', device_id='0'))

# Build a Deeplab v3+ network
image = imread("./simple_images/car_02.jpg")
x = nn.Variable((1, 3, target_h, target_w), need_grad=False)
deeplabv3 = DeepLabV3plus('voc-coco',output_stride=8)
y = deeplabv3(x)

# preprocess image
processed_image = ProcessImage(image, target_h, target_w)
input_array = processed_image.pre_process()

# Compute inference
x.d = input_array
y.forward(clear_buffer=True)
print ("done")
output = np.argmax(y.d, axis=1)

# Apply post processing
post_processed = processed_image.post_process(output[0])

#Display predicted class names
predicted_classes = np.unique(post_processed).astype(int)
for i in range(predicted_classes.shape[0]):
    print('Classes Segmented: ', deeplabv3.category_names[predicted_classes[i]])

# save inference result
processed_image.save_segmentation_image("./output.png")