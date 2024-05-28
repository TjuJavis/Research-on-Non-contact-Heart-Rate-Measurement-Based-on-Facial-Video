import threading
import time
from queue import Queue

import cv2 as cv
import dlib
#import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns

sns.set()

class CAM2FACE:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'shape_predictor_81_face_landmarks.dat')

        # 获取计算机的前置摄像头并获取帧率
        self.cam = cv.VideoCapture(0)
        if not self.cam.isOpened():  # 检查摄像头是否打开
            print('ERROR: 无法打开摄像头。请检查摄像头是否连接并重试。正在退出程序。')
            self.cam.release()
            return

        self.fps = 20

        # 初始化队列
        self.Queue_rawframe = Queue(maxsize=3)
        self.Queue_Sig_left = Queue(maxsize=256)
        self.Queue_Sig_right = Queue(maxsize=256)
        self.Queue_Sig_fore = Queue(maxsize=256)
        self.Queue_Time = Queue(maxsize=64)

        self.Ongoing = False
        self.Flag_face = False
        self.Flag_Queue = False

        self.frame_display = None
        self.face_mask = None

        self.Sig_left, self.Sig_right, self.Sig_fore= None,None,None

    # 初始化进程并启动
    def PROCESS_start(self):
        self.Ongoing = True
        self.capture_process_ = threading.Thread(target=self.capture_process)
        self.roi_cal_process_ = threading.Thread(target=self.roi_cal_process)

        self.capture_process_.start()
        self.roi_cal_process_.start()

    # 捕捉摄像头中的帧并放入队列中
    def capture_process(self):
        while self.Ongoing:
            try:
                self.ret, frame = self.cam.read()
                self.frame_display = copy.copy(frame)

                if self.Queue_Time.full():
                    self.Queue_Time.get_nowait()
                    self.update_fps()

                if not self.ret:
                    self.Ongoing = False
                    break

                if self.Queue_rawframe.full():
                    self.Queue_rawframe.get_nowait()
                else:
                    self.Queue_Time.put_nowait(time.time())

                self.Queue_rawframe.put_nowait(frame)

            except Exception as e:
                pass

    # 从原始帧中计算感兴趣区域
    def roi_cal_process(self):
        while self.Ongoing:
            try:
                frame = self.Queue_rawframe.get_nowait()  # 从原始帧队列中获取帧
            except Exception as e:
                continue

            # 计算感兴趣区域
            ROI_left, ROI_right, ROI_fore = self.ROI(frame)

            if ROI_left is not None and ROI_right is not None and ROI_fore is not None:
                self.hist_left = self.RGB_hist(ROI_left)  # 计算直方图
                self.hist_right = self.RGB_hist(ROI_right)
                self.hist_fore = self.RGB_hist(ROI_fore)

                if self.Queue_Sig_left.full():
                    self.Sig_left = copy.copy(list(self.Queue_Sig_left.queue))
                    self.Queue_Sig_left.get_nowait()  # 移除最早的左侧信号
                else:
                    self.Flag_Queue = False

                if self.Queue_Sig_right.full():
                    self.Sig_right = copy.copy(list(self.Queue_Sig_right.queue))
                    self.Queue_Sig_right.get_nowait()  # 移除最早的右侧信号
                else:
                    self.Flag_Queue = False

                if self.Queue_Sig_fore.full():
                    self.Sig_fore = copy.copy(list(self.Queue_Sig_fore.queue))
                    self.Queue_Sig_fore.get_nowait()  # 移除最早的前部信号
                    self.Flag_Queue = True
                else:
                    self.Flag_Queue = False

                # 将直方图转换为特征值并放入对应的信号队列中
                self.Queue_Sig_left.put_nowait(self.Hist2Feature(self.hist_left))
                self.Queue_Sig_right.put_nowait(self.Hist2Feature(self.hist_right))
                self.Queue_Sig_fore.put_nowait(self.Hist2Feature(self.hist_fore))

            else:
                self.hist_left, self.hist_right, self.hist_fore= None,None,None  # 清空直方图
                self.Queue_Sig_left.queue.clear()  # 清空信号队列
                self.Queue_Sig_right.queue.clear()
                self.Queue_Sig_fore.queue.clear()

    # 获取人脸特征点
    def Marker(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转为灰度图像
        faces = self.detector(img_gray)  # 检测人脸
        if len(faces) == 1:  # 如果检测到一个人脸
            face = faces[0]
            landmarks = [[p.x, p.y] for p in self.predictor(img, face).parts()]
        try:
            return landmarks  # 返回人脸特征点的坐标
        except:
            return None

    # 对图像进行预处理以提高性能
    def preprocess(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)  # 使用高斯模糊进行预处理

    # 绘制感兴趣区域
    def ROI(self, img):
        img = self.preprocess(img)  # 预处理图像
        landmark = self.Marker(img)  # 获取人脸特征点

        cheek_left = [1, 2, 3, 4, 48, 31, 28, 39]
        cheek_right = [15, 14, 14, 12, 54, 35, 28, 42]
        forehead = [69, 70, 71, 80, 72, 25, 24, 23, 22, 21, 20, 19, 18]

        mask_left = np.zeros(img.shape, np.uint8)  # 创建掩码图像
        mask_right = np.zeros(img.shape, np.uint8)
        mask_fore = np.zeros(img.shape, np.uint8)
        mask_display = np.zeros(img.shape, np.uint8)
        try:
            self.Flag_face = True  # 设置检测到人脸的标志为True
            pts_left = np.array([landmark[i] for i in cheek_left], np.int32).reshape((-1, 1, 2))
            pts_right = np.array([landmark[i] for i in cheek_right], np.int32).reshape((-1, 1, 2))
            pts_fore = np.array([landmark[i] for i in forehead], np.int32).reshape((-1, 1, 2))
            mask_left = cv.fillPoly(mask_left, [pts_left], (255, 255, 255))
            mask_right = cv.fillPoly(mask_right, [pts_right], (255, 255, 255))
            mask_fore = cv.fillPoly(mask_fore, [pts_fore], (255, 255, 255))

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 30))
            mask_left = cv.erode(mask_left, kernel=kernel, iterations=1)
            mask_right = cv.erode(mask_right, kernel=kernel, iterations=1)
            mask_fore = cv.erode(mask_fore, kernel=kernel, iterations=1)

            mask_display_left, mask_display_right = copy.copy(mask_left), copy.copy(mask_right)
            mask_display_fore = copy.copy(mask_fore)

            mask_display_left[:, :, 1] = 0
            mask_display_right[:, :, 0] = 0
            mask_display_fore[:, :, 2] = 0

            mask_display = cv.bitwise_or(mask_display_left, mask_display_right)
            mask_display = cv.bitwise_or(mask_display, mask_display_fore)

            self.face_mask = cv.addWeighted(mask_display, 0.25, img, 1, 0)

            ROI_left = cv.bitwise_and(mask_left, img)
            ROI_right = cv.bitwise_and(mask_right, img)
            ROI_fore = cv.bitwise_and(mask_fore, img)

            return ROI_left, ROI_right, ROI_fore
        except Exception as e:
            self.face_mask = img
            self.Flag_face = False
            return None, None, None

    # 计算ROI的直方图
    def RGB_hist(self, roi):
        b_hist = cv.calcHist([roi], [0], None, [256], [0, 256])  # 计算直方图
        g_hist = cv.calcHist([roi], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([roi], [2], None, [256], [0, 256])
        b_hist = np.reshape(b_hist, (256))  # 将直方图展平
        g_hist = np.reshape(g_hist, (256))
        r_hist = np.reshape(r_hist, (256))
        b_hist[0] = 0  # 将直方图中第一个元素设为0
        g_hist[0] = 0
        r_hist[0] = 0
        r_hist = r_hist / np.sum(r_hist)  # 将直方图进行归一化
        g_hist = g_hist / np.sum(g_hist)
        b_hist = b_hist / np.sum(b_hist)
        return [r_hist, g_hist, b_hist]

    # 直方图转换为特征值
    def Hist2Feature(self, hist):
        hist_r = hist[0]
        hist_g = hist[1]
        hist_b = hist[2]

        hist_r /= np.sum(hist_r)
        hist_g /= np.sum(hist_g)
        hist_b /= np.sum(hist_b)

        dens = np.arange(0, 256, 1)
        mean_r = dens.dot(hist_r)
        mean_g = dens.dot(hist_g)
        mean_b = dens.dot(hist_b)

        return [mean_r, mean_g, mean_b]

    # 清理资源
    def __del__(self):
        self.Ongoing = False
        self.cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    cam2roi = CAM2FACE()
    cam2roi.PROCESS_start()
    Hist_left_list = []
    Hist_right_list = []
    while True:
        print(cam2roi.fps)