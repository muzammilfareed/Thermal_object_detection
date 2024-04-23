import cv2
from yolov8 import YOLOv8
import os
from yolov8.utils import draw_text,draw_text_nor



arlam_class_name = ["climbing","crawling","creeping","standing","stooping"]
# arlam_class_name = ["dog","other"]
class_name = ["climbing","crawling","creeping","dog","other","standing","stooping"]

# Initialize yolov8 object detector
model_path = "static/models/thermal_update.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)








# for root,dir,files in os.walk("test"):
#     for file in files:
def main(image_path):
    # image_path = os.path.join(root,file)
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)
    for box,score,clss in zip(boxes, scores, class_ids):
        label = class_name[int(clss)]
        if label in arlam_class_name:
            print(box)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 5)
            caption = f'{label} {int(score * 100)}%'
            draw_text(img, caption, box, (0,0,255), font_size, text_thickness)
        else:
            print(box)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            caption = f'{label} {int(score * 100)}%'
            draw_text_nor(img, caption, box, (0, 255, 255), font_size, text_thickness)

    # cv2.imwrite(f"result/detected_objects{file.split('.')[0]}.jpg", img)
    return img