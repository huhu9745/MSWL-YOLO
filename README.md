# MSWL-YOLO

#As stated in the paper, first set up the required environment.


# run
`val.py` is used to test the accuracy of the validation set in the paper.

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
              device = '0',     #`device` refers to selecting your GPU card.
              )
              

`train.py` is the training program.
    model = YOLO(r'D:\YOLOV8\main\ultralytics\cfg\models\v8\yolov8-bifpn+MASPPF+WISEIOU.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='D:\YOLOV8\main\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD', # using SGD

                # patience=0, # close earlystop
                # resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
 `best.py` contains the best weights from the training in the paper.


# lamp
The LAMP pruning is located in `dataset/Visdrone2021/ultralytics-prune-20240726/ultralytics-prune/compress.py`.

        'model': 'runs/train/yolov10n-visdrone/weights/best.pt',    # Here, you need to specify the weights of the model that was trained earlier.
        'data':'/home/hjj/Desktop/dataset/dataset_visdrone/data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 32,
        'workers': 4,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov10n-visdrone-lamp-exp2',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 2.0,                  #  Here, you need to select the pruning **speed**.
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }



run compress.py





                
