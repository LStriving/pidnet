import numpy as np
import torch
import models
import os
import cv2
from torch.nn import functional as F
from PIL import Image
from scipy.spatial.distance import cdist


def get_palette(num_cls=20):
        """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

class SegAPI(object):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.model = self.init_model()
        self.mean = mean
        self.std = std
        palette = get_palette(num_cls=14)
        self.palette = palette[:3*5]+palette[3*6:3*10]+palette[3*12:3*13]+palette[3*11:3*12]
    
    def input_transform(self, image, bgr=True):
        if bgr:
            image = image.astype(np.float32)[:, :, ::-1] # bgr->rgb
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    
    def preprocess(self, image):
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        return image
    
    def get_pred(self, image):
        img_tensor = self.preprocess(image).cuda()
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = pred[1]
            pred = F.interpolate(
                input=pred, size=img_tensor.shape[-2:],
                mode='bilinear', align_corners=True
            )
            pred_np = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)[0]
        return pred_np
    
    def get_vis(self, image):
        pred = self.get_pred(image)
        color_pred = Image.fromarray(np.asarray(pred, dtype=np.uint8))
        color_pred.putpalette(self.palette)
        color_pred_np = np.array(color_pred.convert("RGB"))

        img_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst_np = cv2.addWeighted(color_pred_np , 0.5, img_np, 0.5, 0)

        pred[pred>0] = 1
        pred_mask = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        out = dst_np * pred_mask + img_np * (1-pred_mask)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
    
    def get_vis_flag(self, image):
        pred = self.get_pred(image)
        # flag
        pred_label = np.unique(pred)
        interact_1_345_flag = '0'
        interact_8_910_flag = '0'
        if 1 in pred_label  and (3 in pred_label or 4 in pred_label  or 5 in pred_label):
            mask_1 = np.where(pred==1, 1, 0).astype(np.uint8)
            mask_3_4_5 = np.where((3<=pred)&(pred<=5),1,0).astype(np.uint8)
            contours_1 = cv2.findContours(mask_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours_345 = cv2.findContours(mask_3_4_5,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            xy_1 = np.concatenate(contours_1[0]).reshape(-1,2)
            xy_345 = np.concatenate(contours_345[0]).reshape(-1,2)
            distances = cdist(xy_1, xy_345)
            if distances.min() < 1.5:
                interact_1_345_flag = '1'
        
        if 8 in pred_label and (9 in pred_label or 10 in pred_label):
            mask_8 = np.where(pred==8, 1, 0).astype(np.uint8)
            mask_9_10 = np.where((9<=pred)&(pred<=10),1,0).astype(np.uint8)
            contours_8 = cv2.findContours(mask_8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            xy_8 = np.concatenate(contours_8[0])
            rect = cv2.minAreaRect(xy_8)
            box = cv2.boxPoints(rect).astype(np.int64)
            mask_rect = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
            mask_rect = cv2.drawContours(mask_rect, [box], 0, 1, cv2.FILLED)
            if np.sum((mask_9_10 & mask_rect)) > 0:
                interact_8_910_flag = '1'

        # vis
        color_pred = Image.fromarray(np.asarray(pred, dtype=np.uint8))
        color_pred.putpalette(self.palette)
        color_pred_np = np.array(color_pred.convert("RGB"))

        img_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst_np = cv2.addWeighted(color_pred_np , 0.5, img_np, 0.5, 0)

        pred[pred>0] = 1
        pred_mask = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        out = dst_np * pred_mask + img_np * (1-pred_mask)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out, interact_1_345_flag, interact_8_910_flag

    def init_model(self):
        # pidnet-s
        model = models.pidnet.PIDNet(m=2, n=3, num_classes=11, planes=32, ppm_planes=96, head_planes=128, augment=True)
        model_state_file = 'best.pt'
        pretrained_dict = torch.load(model_state_file)
        print('load model from {}'.format(model_state_file))
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        device = torch.device('cuda')
        model.eval()
        model.to(device)

        return model

if __name__ == '__main__':
    api = SegAPI()
    path = '/mnt/cephfs/home/zhoukai/Codes/PIDNet/api_test.jpg'
    image = cv2.imread(path)
    output, flag_1_345, flag_8_910 = api.get_vis_flag(image)
    cv2.imwrite('./api_output.jpg', output)
    print(flag_1_345)
    print(flag_8_910)

    


