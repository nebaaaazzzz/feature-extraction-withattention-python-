
from torchvision import models
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image,show_factorization_on_image,preprocess_image

model=models.resnet50(weights="DEFAULT")
model.eval()


# fix target class label (of the Imagenet class of interest!)

targets = [ClassifierOutputTarget(9)] 

# fix the target layer (after which we'd like to generate the CAM)
target_layers = [model.layer1]

# instantiate the model
# if type =='gradcam':
cam = GradCAM(model=model, target_layers=target_layers)
# else:
#     cam = GradCAMPlusPlus(model=model, target_layers=target_layers) # use this for Grad-CAM++

# Preprocess input image, get the input image tensor
img = np.array(PIL.Image.open('images (1).jpeg'))
img = cv2.resize(img, (300,300))
img = np.float32(img) / 255
input_tensor = preprocess_image(img)

# generate CAM
grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])

# display the original image & the associated CAM
images = np.hstack((np.uint8(255*img), cam_image))
plt.imshow(images)
plt.show()