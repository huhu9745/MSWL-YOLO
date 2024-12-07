import cv2
import subprocess
from ultralytics import YOLO

# 载入 YOLOv8 模型
model = YOLO('model/yolov8n.pt')

# 获取视频内容
cap = cv2.VideoCapture("0")

# 获取原视频的宽度和高度
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置 FFmpeg 子进程，用于推流
rtmp_url = 'rtmp://127.0.0.1:1935/live/stream'  # 修改为您的 NGINX RTMP 服务器地址
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(original_width, original_height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp_url]

# 启动 FFmpeg 进程
proc = subprocess.Popen(command, stdin=subprocess.PIPE)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 对帧运行 YOLOv8 推理
        results = model(frame)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 将处理后的帧写入 FFmpeg 进程
        proc.stdin.write(annotated_frame.tobytes())

        # 显示带有标注的帧
        cv2.imshow("YOLOv8 推理", annotated_frame)

        # 如果按下 'q' 键，则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放视频捕获对象
cap.release()

# 关闭 FFmpeg 进程
proc.stdin.close()
proc.wait()

# 关闭显示窗口
cv2.destroyAllWindows()