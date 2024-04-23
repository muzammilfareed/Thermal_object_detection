import cv2
from yolov8.utils import draw_text,draw_text_nor

from yolov8 import YOLOv8
arlam_class_name = ["climbing","crawling","creeping","standing","stooping"]
# arlam_class_name = ["dog","other"]
class_name = ["climbing","crawling","creeping","dog","other","standing","stooping"]
# Initialize the webcam

cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "static/models/thermal_update.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break
    img_height, img_width = frame.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # combined_img = yolov8_detector.draw_detections(frame)
    for box, score, clss in zip(boxes, scores, class_ids):
        label = class_name[int(clss)]
        if label in arlam_class_name:
            print(box)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            caption = f'{label} {int(score * 100)}%'
            draw_text(frame, caption, box, (0, 0, 255), font_size, text_thickness)
        else:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            caption = f'{label} {int(score * 100)}%'
            draw_text_nor(frame, caption, box, (0, 0, 255), font_size, text_thickness)

    cv2.imshow("Detected Objects", frame)


    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
