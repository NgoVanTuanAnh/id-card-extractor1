from PIL import Image
from src.crnn.train import Train
from src.crnn.utils import *
from src.crnn.config import *
from src.detecto.detecto import core
from src.utils import *
from torchvision.models import resnet18
from torchvision import transforms
from src.crnn.models import CRNN
import gradio as gr
import numpy as np
import cv2
import yaml
import os
import sys
sys.path.append('src/detecto/')
sys.path.append('src/crnn/')


class Inference:

    def __init__(self, config):
        resnet = resnet18()
        self.cfg = config
        self.corner = self._load(self.cfg.DETECTO.CORNER)
        self.info_ext = self._load(self.cfg.DETECTO.INFO)
        self.crnn = Train(self.cfg.CRNN.TRAINING)
        self.crnn.model = CRNN(backbone=resnet,
                               num_chars=len(self.cfg.CRNN.VOCAB),
                               rnn_hidden_size=self.cfg.CRNN.RNN_HIDDEN)
        self.crnn.load(self.cfg.CRNN.CHECKPOINT)

    def _load(self, cfg):
        model = core.Model(cfg.CLASSES)
        return model.load(cfg.CHECKPOINT, cfg.CLASSES)

    def _get_point(self, image, model):
        '''
        Tìm các tọa độ điểm chứa các (bouding box) và nhãn của chúng 
        '''
        # Lấy nhãn và các boxes
        labels, boxes, _ = model.predict_top(image)
        # Áp dụng thuật toán nms để loại bỏ các bbox dư thừa
        final_boxes, final_labels = non_max_suppression_fast(boxes=boxes.numpy(),
                                                             labels=labels,
                                                             overlapThresh=0.15)
        return final_boxes, final_labels

    def _getTransform(self, image):
        """_summary_:
        Dựa vào thông tin các điểm tọa độ để transform ảnh ban đầu 
        thành ảnh đã được cắt theo center của bbox

        Args:
            image (PIL.Image): Ảnh ban đầu
            model (core.Model): Model detect cạnh

        Returns:
            nd.array: Ảnh đã được crop
        """
        boxes, labels = self._get_point(image, self.corner)

        # Get các điểm trung tâm của boxes
        final_points = list(map(get_center_point, boxes))
        label_boxes = dict(zip(labels, final_points))

        # Tìm các cạnh còn thiếu
        corner_missed = [key for key in self.cfg.DETECTO.CORNER.CLASSES if key not in list(
            label_boxes.keys())]
        if corner_missed != []:
            missed = corner_missed[0]
            label_boxes[missed] = find_miss_corner(missed, label_boxes)
        # Gộp các điểm trung tâm thành 1 bbox
        source_points = np.float32([
            label_boxes['top_left'], label_boxes['top_right'],
            label_boxes['bottom_right'], label_boxes['bottom_left']
        ])

        # Transform
        crop = perspective_transform(image, source_points)
        return crop

    def _get_info(self, image):
        """_summary_:
        Get information from id card by using OCR and DNN.

        Args:
            image (ndarray): Ảnh đã được crop từ getTransform
            model (core.Model): Model detect info

        Returns:
            image (np.ndarray): Ảnh crop nhưng đã được visualize 
            info (dict): Thông tin đã được predict 
        """
        # Get bboxes, and labels
        boxes, labels = self._get_point(image, self.info_ext)

        # Khởi tạo thông tin cần lưu
        info = {
            'id': '',
            'name': '',
            'date': ''
        }

        # Sử dụng model VietOCR dự đoán văn bản trong bbox
        for _, batch in enumerate(zip(boxes, labels)):
            bbox, label = batch
            # Get tọa độ của box
            x, y, w, h = bbox
            # Get ảnh trong bbox
            img = Image.fromarray(image[y:h, x-5:w+5])
            # Cho ảnh văn bản qua mô hình VietOCR
            pred = self._image_to_text(img)
            # Lưu lại thông tin
            info[label] = pred
            # Visualize
            cv2.rectangle(image, (x-5, y), (w+5, h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        return image, info

    def _image_to_text(self, image):
        cfg = self.cfg.CRNN
        transform = transforms.Compose([
            transforms.Pad(padding=cfg.PAD.PADDING,
                           fill=cfg.PAD.FILL),
            transforms.ToTensor(),
            transforms.Resize(size=(cfg.HEIGHT, cfg.WIDTH)),
            transforms.Normalize(cfg.NORMLIZE.MEAN, cfg.NORMLIZE.STD)
        ])
        image = transform(image)
        text_batch_logits = self.crnn.model(image[None, :, :, :])
        decode = decode_predictions(text_batch_logits.cpu())
        return correct_prediction(decode[0])

    def __call__(self, image):
        image_crop = self._getTransform(image)
        image_crop_copy = image_crop.copy()
        image_info, info = self._get_info(image_crop_copy)
        return image_crop, image_info, info['id'], info['name'], info['date']


def nested_parser(data):
    # Base case: if data is not a dictionary, return it as is
    if not isinstance(data, dict):
        return data
    # Recursive case: if data is a dictionary, create a class object with attributes
    else:
        class Config:
            pass
        for key, value in data.items():
            setattr(Config, key, nested_parser(value))
        return Config


if __name__ == "__main__":
    # Load file config
    with open('config.yml', 'r') as fp:
        config = yaml.safe_load(fp)

    config = nested_parser(config)
    infer = Inference(config)
    demo = gr.Interface(infer,
                        inputs=['image'],
                        outputs=[gr.Image(label='Image Crop'),
                                 gr.Image(label='Detect Info'),
                                 gr.Textbox(label='ID'),
                                 gr.Textbox(label='Name'),
                                 gr.Textbox(label='Date of Birth')])
    demo.launch(share=False)
