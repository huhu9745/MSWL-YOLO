import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\YOLOV8\main\best.pt')
    # D:\YOLOV8\ultralytics - 20240831\ultralytics - main\dataset\VOCdevkit\local_data.yaml
    #/opt/data/private/HuFengXiang/dataset/data.yaml
    model.val(data=r'D:\YOLOV8\main\dataset\data.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              device = '0',
              )