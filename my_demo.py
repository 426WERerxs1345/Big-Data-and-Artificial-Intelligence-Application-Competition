# -*- coding: utf-8 -*-
'''
作者：智能车之家
时间：2020/3/20
'''
import _thread
from aip import AipSpeech
import sys
import time
import requests,json
import base64
import signal
import sys
import os
import urllib3
from MyEncoder import MyEncoder
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import pyttsx3
import os,time,random
from threading import Thread
from queue import Queue

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pyrealsense2 as rs       #realsense的sdk模块
from models.experimental import attempt_load
#加载一个模型集合的权重=[a,b,c]或单个模型的权重=[a]或权重=a

from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)   
urllib3.disable_warnings()
APP_ID = '37887761'
API_KEY = 'eliFqeawL1w6feaPIFehbq5G'
SECRET_KEY = '2bodsLv0GCfxzNiPv3QZ2eEGKSf9Age3'
ALLMSG = ""
interrupted = False

#---------------------------------------------------------------------------#
#获取百度云token    
def getaccess_token():
    host='https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=eliFqeawL1w6feaPIFehbq5G&client_secret=2bodsLv0GCfxzNiPv3QZ2eEGKSf9Age3'
    header_1 = {'Content-Type':'application/json; charset=UTF-8'}
    request=requests.post(host,headers =header_1)
    access_token=request.json()['access_token']
    print(access_token)
    return access_token
#在线语音识别  与逻辑处理
def Speech(access_token):
    global detector
    global flage
    #string = "你是不是傻".encode('utf-8')
    request_url = "http://vop.baidu.com/server_api"
    headers = { 'Content-Type' : 'application/json' }
    VOICE_RATE = 16000
    WAVE_FILE = "ddd.wav" 
    USER_ID = "zhp-fw" 
    WAVE_TYPE = "wav"
    # begin Speech
    os.system('aplay /home/pi/snowboy/dong.wav')
    #os.system('espeak -vzh "%s"'%"你是不是傻".encode('utf-8'))
    os.system('arecord -d 4 -r 16000 -c 1 -t wav -f S16_LE  ddd.wav')# -D plughw:1,0 加上这个可能就会出现繁忙警告
    #os.system('rec -r 16000 -c 1 -b 4 -e signed-integer ddd.wav')
    f = open(WAVE_FILE, "rb") 
    speech = base64.b64encode(f.read())
    size = os.path.getsize(WAVE_FILE)
    data_json = json.dumps({"format":WAVE_TYPE, "rate":VOICE_RATE, "channel":1,"cuid":USER_ID,"token":access_token,"speech":speech,"len":size}, cls=MyEncoder, indent=4)
    #request_url = request_url + "?access_token=" + access_token
    response = requests.post(request_url, data=data_json,headers=headers)
    print(response.text)
    if response.json().get('err_msg')=='success.':
        word = response.json().get('result')[0]#.encode('utf-8')
        return word
#百度语音合成
def baidu_tts(words):
    result = client.synthesis(text = words, options={'vol': 5})
#     print(result)
    if not isinstance(result, dict):
        with open('audio.mp3', 'wb') as f:
            f.write(result)
        os.system('mplayer audio.mp3')
    else:
        print(result)
def detect(myqueue: Queue,save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference运行推理
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img (，c，w，h)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
                # 更换浮点数：float16（https://blog.csdn.net/leo0308/article/details/117398166）
    pipeline = rs.pipeline()        # 声明realsense对象（https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.pipeline.html）
                                    # 类抽象了摄像机配置和流，以及视觉模块的触发和线程。
                                    # 它让应用程序专注于计算机视觉模块的输出，或者设备输出的数据。
                                    # 该流水线可以对计算机视觉模块进行管理，并将其作为一个处理模块来实现。
    # 创建 config 对象：
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #rgb流
            # 重载函数，配置特定的流
            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.config.html?highlight=config

    # Start streaming
    pipeline.start(config)  # 根据配置启动数据流
    align_to_color = rs.align(rs.stream.color)      #realsense负责流对齐的 SDK 类
                        # 默认是深度流对齐到align_to参数；
                        # 其他流对齐到深度流，align_to设置为RS2_STREAM_DEPTH
                        # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.align.html?highlight=align#pyrealsense2.align

    while True:
        baidu_tts("请说出目标物体")
        mubiao_name = Speech(access_token)

        start = time.time()
        # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
        frames = pipeline.wait_for_frames() # 等待双帧同步，该方法阻塞调用线程，并获取最新的未读帧集；
                                            # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.pipeline.html?highlight=wait_for_frames#pyrealsense2.pipeline.wait_for_frames
        frames = align_to_color.process(frames)
        # depth_frame = frames.get_depth_frame()
        depth_frame = frames.get_depth_frame()  # 检索深度帧返回实例
        color_frame = frames.get_color_frame()  # 同上
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # 设置掩码阈值
        mask = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.uint8)
        mask[0:480, 320:640] = 255

        sources = [source]  # 数据源
        imgs = [None]
        path = sources
        imgs[0] = color_image   # rgb流数据
        im0s = imgs.copy()
        img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]  #调整每张图片大小为32像素多矩形
        img = np.stack(img, 0)  #将img沿着0轴打包
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
        img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)   #返回一个连续的array，其内存是连续的。
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        if pred == [None]:
            print('跳转到点云')
            # -*- coding: utf-8 -*-
            #os.system('cd rgbd_ground_filter_ws/')
            #os.system('source ./devel/setup.bash')
            #os.system('roslaunch rgbd_ground_filter rgbd_ground_filter.launch')
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                    distance_list = []
                    mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2), int((int(xyxy[1]) + int(xyxy[3])) / 2)]  # 确定索引深度的中心像素位置左上角和右下角相加在/2
                    min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))  # 确定深度搜索范围
                    # print(box,)
                    randnum = 40
                    for i in range(randnum):
                        bias = random.randint(-min_val // 4, min_val // 4)
                        dist = depth_frame.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))     #获取预测结果点的深度信息
                        # print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
                        if dist:
                            distance_list.append(dist)
                    distance_list = np.array(distance_list)
                    distance_list = np.sort(distance_list)[
                                    randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波

                    label = '%s %.2f%s' % (names[int(cls)], np.mean(distance_list), 'm')
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    #names[int(cls)]代表灯，np.mean(distance_list)代表距离，'m'代表单位
                    print('%s %.2f%s' % (names[int(cls)], np.mean(distance_list), 'm'))

                    state = names[int(cls)]
                    distance = float(np.mean(distance_list))
                    q1.put((state,distance))
                    if start == 'bottle' and '瓶子'  in mubiao_name:
                        baidu_tts('识别成功，物体为瓶子')
                        baidu_tts('检测当前距离为'+str(distance))
                    elif start == 'cup' and '水杯' in mubiao_name:
                        baidu_tts('识别成功，物体为水杯')
                        baidu_tts('检测当前距离为'+str(distance))
                    elif start == 'knife' and '刀'  in mubiao_name:
                        baidu_tts('识别成功，物体为小刀')
                        baidu_tts('检测当前距离为' + str(distance))
                    else:
                        baidu_tts('检测物体与语音播报的不相符')
                    time.sleep(0.1)
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
    print('Done. (%.3fs)' % (time.time() - t0))
if __name__ == '__main__':

    access_token = getaccess_token()
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images',
                        help='source')  # file/folder, 0 for webcam#inference/images这里可以选择realsense还是自己的电脑
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)
    q1 = Queue(1000)
    p1 = Thread(target=detect, name='目标检测', args=(q1,))
    # 启动子进程，写入
    p1.start()

