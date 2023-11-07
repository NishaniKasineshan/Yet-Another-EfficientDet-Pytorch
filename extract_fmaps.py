"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
# from PIL import Image
import torchvision.transforms as T

from new_backbone import EfficientDetBackbone
import matplotlib.pyplot as plt
import cv2

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


torch.cuda.empty_cache()
compound_coef = 0
force_input_size = None  # set None to use default size
img_path = 'datasets/shape/val/975.jpg'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

# obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
#             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#             'toothbrush']

# obj_list=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

obj_list=['rectangle','circle']

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
        print("inside")

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            print("working")
            # cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])
            cv2.imwrite(f'test/eff_img{i}.jpg',imgs[i])

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
# model.load_state_dict(torch.load('logs/shape/regnet-d0_40_0.pth', map_location='cpu'))
# model.requires_grad_(False)
# model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)
    # p3,p4,p5=model(x)
    
    # features=model(x)
    print(classification.shape)
    print(regression.shape)
    print(anchors.shape)
    
    p3_bifpn,p4_bifpn,p5_bifpn,p6_bifpn,p7_bifpn=features
    p3_bifpn_tensor=p3_bifpn.cpu()
    p3_bifpn_array=p3_bifpn_tensor.detach().numpy()
    print(p3_bifpn_array.shape)
    # combined_image_p3_bifpn = p3_bifpn_array[0].sum(axis=0)
    # plt.imsave('combined_image_p3_regnet_bifpn.png', combined_image_p3_bifpn, cmap='viridis', format='png')
    
    p4_bifpn_tensor=p4_bifpn.cpu()
    p4_bifpn_array=p4_bifpn_tensor.detach().numpy()
    print(p4_bifpn_array.shape)
    # combined_image_p4_bifpn = p4_bifpn_array[0].sum(axis=0)
    # plt.imsave('combined_image_p4_regnet_bifpn.png', combined_image_p4_bifpn, cmap='viridis', format='png')
    
    p5_bifpn_tensor=p5_bifpn.cpu()
    p5_bifpn_array=p5_bifpn_tensor.detach().numpy()
    print(p5_bifpn_array.shape)
    # combined_image_p5_bifpn = p5_bifpn_array[0].sum(axis=0)
    # plt.imsave('combined_image_p5_regnet_bifpn.png', combined_image_p5_bifpn, cmap='viridis', format='png')
    
    p6_bifpn_tensor=p6_bifpn.cpu()
    p6_bifpn_array=p6_bifpn_tensor.detach().numpy()
    print(p6_bifpn_array.shape)
    # combined_image_p6_bifpn = p6_bifpn_array[0].sum(axis=0)
    # plt.imsave('combined_image_p6_regnet_bifpn.png', combined_image_p6_bifpn, cmap='viridis', format='png')
    
    p7_bifpn_tensor=p7_bifpn.cpu()
    p7_bifpn_array=p7_bifpn_tensor.detach().numpy()
    print(p7_bifpn_array.shape)
    # combined_image_p7_bifpn = p7_bifpn_array[0].sum(axis=0)
    # plt.imsave('combined_image_p7_regnet_bifpn.png', combined_image_p7_bifpn, cmap='viridis', format='png')

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    print("finished assign")
    # print(p3)
    # print(p3.type())
    # tensor_array_p3 = p3.cpu()
    # numpy_array_p3 = tensor_array_p3.detach().numpy()
    # print(numpy_array_p3.shape)
    # normalized_array = ((numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255).astype('uint8')
    # # print(normalized_array)
    # image = Image.fromarray(normalized_array.squeeze())
    # print(numpy_array)
    # numpy_array = ((numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255).astype('uint8')
    # cv2.imwrite('p3.jpg',numpy_array)
    # combined_image_p3 = numpy_array_p3[0].sum(axis=0)  # Combine all channels

    # Create an image with the viridis colormap
    # plt.imsave('combined_image_p3.png', combined_image_p3, cmap='viridis', format='png')
    
    # print(p4.type())
    # tensor_array_p4 = p4.cpu()
    # numpy_array_p4 = tensor_array_p4.detach().numpy()
    # print(numpy_array_p4.shape)
    # combined_image_p4 = numpy_array_p4[0].sum(axis=0)
    # plt.imsave('combined_image_p4.png', combined_image_p4, cmap='viridis', format='png')
    
    
    # print(p5.type())
    # tensor_array_p5 = p5.cpu()
    # numpy_array_p5 = tensor_array_p5.detach().numpy()
    # print(numpy_array_p5.shape)
    # combined_image_p5 = numpy_array_p5[0].sum(axis=0)
    # plt.imsave('combined_image_p5.png', combined_image_p5, cmap='viridis', format='png')


    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)
# print(out)
out = invert_affine(framed_metas, out)
print(out)
display(out, ori_imgs, imshow=False, imwrite=True)

# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

  