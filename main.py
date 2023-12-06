import sys
sys.path.append('src/detecto/')
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.predict import Predictor
from src.detecto.detecto import core, utils
from src.utils import non_max_suppression_fast, get_center_point, perspective_transform, find_miss_corner

def load_model_id_card():
    classes = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    model = core.Model(classes)
    return model.load('./src/weight/id_card_4_corner.pth', classes)

def load_model_detect_info():
    classes = ['id', 'name', 'date']
    model = core.Model(classes)
    return model.load('./src/weight/detect_info_.pth', classes)

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

def get_info(image, detector, model):
    boxes, labels = get_point(image, model)
    info = {}
    for batch in zip(boxes, labels):
        bbox, label = batch
        x, y, w, h = bbox
        img = Image.fromarray(image[y:h, x-5:w+5])
        anno = detector.predict(img)
        info[label] = anno
    return info
    
def main(image):
    image_crop = getTransform(image, model_corner)
    info = get_info(image_crop, detector, model_info)
    return info['id'], info['name'], info['date']

def GUI():
    demo = gr.Interface(main, 
                        inputs=['image'], 
                        outputs=[gr.Textbox(label='ID'),
                                 gr.Textbox(label='Name'),
                                 gr.Textbox(label='Date of Birth')])
    demo.launch(share=False)

if __name__ == "__main__":
    model_corner = load_model_id_card()
    model_info = load_model_detect_info()
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    detector = Predictor(config)
    GUI()