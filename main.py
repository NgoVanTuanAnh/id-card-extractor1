import os
import cv2
import sys
sys.path.append('src/detecto/')
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from src.detecto.detecto import core, utils
from src.utils import non_max_suppression_fast, get_center_point, perspective_transform, find_miss_corner
import datetime

def load_model_id_card():
    classes = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    model = core.Model(classes)
    return model.load('./src/weight/weight-model-13.pth', classes)

def load_model_detect_info():
    classes = ['id', 'name', 'date']
    model = core.Model(classes)
    return model.load('./src/weight/detect-id/best-model.pth', classes)

def get_point(image, model):
    labels, boxes, scores = model.predict_top(image)
    final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.15)
    return final_boxes, final_labels

def getTransform(image, model):
    boxes, labels = get_point(image, model)
    final_points = list(map(get_center_point, boxes))
    label_boxes = dict(zip(labels, final_points))
    corner_missed = [key for key in ['top_left', 'top_right', 'bottom_right', 'bottom_left'] if key not in list(label_boxes.keys())]
    if corner_missed != []:
        missed = corner_missed[0]
        label_boxes[missed] = find_miss_corner(missed, label_boxes)

    source_points = np.float32([
        label_boxes['top_left'], label_boxes['top_right'], 
        label_boxes['bottom_right'], label_boxes['bottom_left']
    ])
        
    # Transform 
    crop = perspective_transform(image, source_points)
    return crop

# def get_info(image, image_, model, save=False):
#     boxes, labels = get_point(image, model)
#     info = {
#         'id': '',
#         'name': '',
#         'date': ''
#     }

#     now = "{:%m-%d-%Y-%H-%M-%S}".format(datetime.datetime.now())
#     dirname = 'id-card_' + now
#     path = os.path.join('./testset', dirname)
#     if not os.path.isdir(path):
#         os.mkdir(path)
#     Image.fromarray(image_).save(os.path.join(path, '0.jpg'))
#     for idx, batch in enumerate(zip(boxes, labels)):
#         bbox, label = batch
#         x, y, w, h = bbox
#         img = Image.fromarray(image[y:h, x-5:w+5])
#         if save:
#             img.save(os.path.join(path, f'{idx+1}.jpg'))
#         # cv2.rectangle(image,(x, y),(w,h),(0,255,0),2)
#         # cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#     return image, info

def get_info(image, image_, model, save=False):
    boxes, labels = get_point(image, model)
    info = {
        'id': '',
        'name': '',
        'date': ''
    }
    for idx, batch in enumerate(zip(boxes, labels)):
        bbox, label = batch
        x, y, w, h = bbox
        img = Image.fromarray(image[y:h, x-5:w+5])
        pred = detector.predict(img)
        info[label] = pred
        cv2.rectangle(image, (x-5, y),(w+5,h),(0,255,0),2)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
    return image, info

def main(image, save_rec=True):
    image_crop = getTransform(image, model_corner)
    image_crop1 = image_crop.copy()
    image_info, info = get_info(image_crop1, image, model_info, save_rec)
    return image_crop, image_info, info['id'], info['name'], info['date']

def GUI():
    demo = gr.Interface(main, 
                        inputs=['image'], 
                        outputs=[gr.Image(label='Image Crop'),
                                 gr.Image(label='Detect Info'),
                                 gr.Textbox(label='ID'),
                                 gr.Textbox(label='Name'),
                                 gr.Textbox(label='Date of Birth')])
    demo.launch(share=False)

if __name__ == "__main__":
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    detector = Predictor(config)
    
    model_corner = load_model_id_card()
    model_info = load_model_detect_info()
    GUI()