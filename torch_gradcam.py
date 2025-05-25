import torch
from torch import nn
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
from pytorch_grad_cam import  GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image,show_factorization_on_image,preprocess_image
from os import listdir
import os
from resnet import get_resnet18

num_classes = 200
image_id = 999
def get_class() :
    with open("data/tiny-imagenet-200/wnids.txt") as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

    images = []    

    dir_path = "data/tiny-imagenet-200/val"    
    imgs_path = os.path.join(dir_path, 'images')
    imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

    with open(imgs_annotations) as r:
        data_info = map(lambda s: s.split('\t'), r.readlines())

    cls_map = {line_data[0]: line_data[1] for line_data in data_info}

    for imgname in sorted(os.listdir(imgs_path)):
        path = os.path.join(imgs_path, imgname)
        item = (path, class_to_idx[cls_map[imgname]])
        images.append(item)


    result = filter(lambda x: x[0] == f"data/tiny-imagenet-200/val/images/val_{image_id}.JPEG", images)

    filtered_result = list(result)
    print(filtered_result)
    return filtered_result[0][1]



config_list =[
    {"attention_type": "CA", "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None , "label" : "Coordinate Attention"},
    {"attention_type": "ECA", "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None , "label" : "Efficient Channel Attention"},
    {"attention_type": "ELA", "ela_kernelsize": 5, "ela_group_setting": "channel/8", "ela_numgroup":16 , "label" :"Efficient Latent Attention - Small"  }, #small
    {"attention_type": "ELA", "ela_kernelsize": 5, "ela_group_setting": "channel", "ela_numgroup": 32 , "label" :"Efficient Latent Attention - Tiny"}, #itny
    {"attention_type": "ELA", "ela_kernelsize": 7, "ela_group_setting": "channel/8", "ela_numgroup": 16 , "label" :"Efficient Latent Attention - large"}, #large
    {"attention_type": "SE", "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None , "label" :"Squeeze and Excitation"},
    {"attention_type": None, "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None, "label": "Baseline"},
    {"attention_type": "SHUFFLE", "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None , "label": "ShuffleNet"},
    {"attention_type": "TRIPLATE", "ela_kernelsize": None, "ela_group_setting": None, "ela_numgroup": None, "label": "Triplet Attention"},
    # {"attention_type": "ELA", "ela_kernelsize": 7, "ela_group_setting": "channel", "ela_numgroup": 16}, #big
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i, file in enumerate(sorted(listdir('./models'))):
    with open(f'./models/{file}/checkpoint_best.pth', 'rb') as f:
        checkpoint = torch.load(f, map_location=device , weights_only=False)
        model = get_resnet18(
            attention_type=config_list[i]['attention_type'],
            ela_kernelsize=config_list[i]['ela_kernelsize'],
            ela_group_setting=config_list[i]['ela_group_setting'],
            ela_numgroup=config_list[i]['ela_numgroup'],
            num_classes=num_classes
        )
        # for key, value in checkpoint["model"].items():
        #     print(f"{key}: {value.shape}")
        
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        model.load_state_dict(checkpoint['model'] )
        model.eval()
        
        targets = [ClassifierOutputTarget(get_class())] 

        # fix the target layer (after which we'd like to generate the CAM)
        target_layers = [model.layer4]

        # instantiate the model
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

        # Preprocess input image, get the input image tensor
        img = np.array(PIL.Image.open(f'data/tiny-imagenet-200/val/images/val_{image_id}.JPEG'))
        img = cv2.resize(img, (300,300))
        img = np.float32(img) / 255
        input_tensor = preprocess_image(img)
        result = model(input_tensor)
        predicted_class = torch.argmax(result, dim=1)
        # print(predicted_class.item())
        #GET THE MAX PREDICTION CLASSj
        
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"{config_list[i]['label']} Total number of parameters: {total_params}")
        # generate CAM
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])

        # # # display the original image & the associated CAM
        images = np.hstack((np.uint8(255*img), cam_image))
        plt.title(f"{config_list[i]['label']} - Grad-CAM++")
        plt.imshow(images)
        plt.show()

