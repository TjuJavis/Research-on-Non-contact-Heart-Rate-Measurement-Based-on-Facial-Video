
import sys
from mainwindow import Ui_MainWindow
#from obspy.signal.detrend import spline
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#import pywt
#from PyEMD import EEMD
import pyqtgraph as pg
#from scipy.interpolate import splrep, splev
from scipy import signal
import numpy as np
import cv2 as cv
from series2rPPG import Series2rPPG

MIN_HZ = 0.83       # 50 BPM - minimum allowed heart rate 
MAX_HZ = 2.5       # 150 BPM - maximum allowed heart rate

class mainwin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainwin, self).__init__(parent)
        self.setupUi(self)

        self.label_fore = QLabel(self.verticalLayoutWidget)
        self.label_left = QLabel(self.verticalLayoutWidget)
        self.label_right = QLabel(self.verticalLayoutWidget)

        self.Signal_fore = pg.PlotWidget(self)
        self.Signal_left = pg.PlotWidget(self)
        self.Signal_right = pg.PlotWidget(self)

        self.Sig_f = self.Signal_fore.plot()
        self.Sig_l = self.Signal_left.plot()
        self.Sig_r = self.Signal_right.plot()

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)

        self.label_fore.setFont(font)
        self.label_fore.setText("Fore Cheek Signal")
        self.Layout_BVP.addWidget(self.label_fore)
        self.Layout_BVP.addWidget(self.Signal_fore)

        self.label_left.setFont(font)
        self.label_left.setText("Left Cheek Signal")
        self.Layout_BVP.addWidget(self.label_left)
        self.Layout_BVP.addWidget(self.Signal_left)

        self.label_right.setFont(font)
        self.label_right.setText("Right Cheek Signal")
        self.Layout_BVP.addWidget(self.label_right)
        self.Layout_BVP.addWidget(self.Signal_right)

        self.face.setScaledContents(True)
        self.processor = Series2rPPG()
        self.processor.PROCESS_start()

        self.TIMER_Frame = QTimer()
        self.TIMER_Frame.setInterval(100)
        self.TIMER_Frame.start()

        self.TIMER_Hist = QTimer()
        self.TIMER_Hist.setInterval(100)
        self.TIMER_Hist.start()

        self.TIMER_SIGNAL = QTimer()
        self.TIMER_SIGNAL.setInterval(100)
        self.TIMER_SIGNAL.start()

        self.bpm_fore = 60
        self.bpm_left = 60
        self.bpm_right = 60

        self.bpm_avg = 60
        self.ModeDict = {'GREEN': self.processor.GREEN,
                         'GREEN-RED': self.processor.GREEN_RED, 'CHROM': self.processor.CHROM,
                         'PBV': self.processor.PBV}
        self.Mode = self.ModeDict['GREEN']
        self.Data_ShowRaw = True
        self.slot_init()

    def slot_init(self):
        self.TIMER_Frame.timeout.connect(self.DisplayImage)
        self.TIMER_SIGNAL.timeout.connect(self.DisplaySignal)
        self.comboBox.activated[str].connect(self.Button_ChangeMode)
        self.Button_RawTrue.clicked.connect(self.Button_Data_RawTrue)
        self.Button_RawFalse.clicked.connect(self.Button_Data_RawFalse)

    def Button_ChangeMode(self, str):
        self.Mode = self.ModeDict[str]

    def Button_Data_RawTrue(self):
        self.Data_ShowRaw = True

    def Button_Data_RawFalse(self):
        self.Data_ShowRaw = False

    def DisplayImage(self):
        Mask = self.processor.series_class.face_mask
        Mask = cv.ellipse(Mask, [320, 240], [80, 120], 0, 0, 360,
                          [0, 255, 0], 1, cv.LINE_AA)
        Mask = cv.circle(Mask, [320, 240], 2, [255, 0, 0], 2, cv.LINE_AA)

        if Mask is not None:

            img = cv.cvtColor(Mask, cv.COLOR_BGR2RGB)
            qimg = QImage(
                img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)

            self.face.setPixmap(QPixmap.fromImage(qimg))


    def DisplayHist(self):
        Hist_fore = np.array(self.processor.series_class.hist_fore)
        Hist_left = np.array(self.processor.series_class.hist_left)
        Hist_right = np.array(self.processor.series_class.hist_right)
        if Hist_fore.size != 1:
            self.Hist_fore_r.setData(Hist_fore[0, :], pen=(255, 0, 0))
            self.Hist_fore_g.setData(Hist_fore[1, :], pen=(0, 255, 0))
            self.Hist_fore_b.setData(Hist_fore[2, :], pen=(0, 0, 255))
        else:
            self.Hist_fore_r.clear()
            self.Hist_fore_g.clear()
            self.Hist_fore_b.clear()
        if Hist_left.size != 1:
            self.Hist_left_r.setData(Hist_left[0, :], pen=(255, 0, 0))
            self.Hist_left_g.setData(Hist_left[1, :], pen=(0, 255, 0))
            self.Hist_left_b.setData(Hist_left[2, :], pen=(0, 0, 255))
        else:
            self.Hist_left_r.clear()
            self.Hist_left_g.clear()
            self.Hist_left_b.clear()
        if Hist_fore.size != 1:
            self.Hist_right_r.setData(Hist_right[0, :], pen=(255, 0, 0))
            self.Hist_right_g.setData(Hist_right[1, :], pen=(0, 255, 0))
            self.Hist_right_b.setData(Hist_right[2, :], pen=(0, 0, 255))
        else:
            self.Hist_right_r.clear()
            self.Hist_right_g.clear()
            self.Hist_right_b.clear()

    # Creates the specified Butterworth filter and applies it.
    def butterworth_filter(self, data, low, high, sample_rate, order=7):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate

        #增加去趋势滤波detrend
        #dspline = 8
        #knots = np.arange(dspline // 2, len(data) - dspline // 2 + 2)
        #tck = splrep(knots, data[knots], k=3)
        #detrended_data = splev(np.arange(len(data)), tck)

        #增加滑动平均
        window_size = 5
        window = np.ones(window_size) / window_size
        smoothed_data = np.convolve(data, window, mode='same')

        #带通滤波
        b, a = signal.butter(order, [low, high], btype='band')
        filted_signal = signal.lfilter(b, a,  smoothed_data)
        """
        pyqtgraph opengl 
        # EEMD分解
        eemd = EEMD()
        eemd_decomposition = eemd(filted_signal)
        imfs = eemd_decomposition[:-1]  # 通常最后一个IMF是噪声，所以这里省略掉

        # 合并所有IMF为一个长向量并计算其频谱
        merged_imfs = np.concatenate(imfs)
        merged_spectra = np.abs(np.fft.fftshift(np.fft.fft(merged_imfs)))

        # 分割总频谱，获取每个IMF的频谱
        imfs_len = [len(imf) for imf in imfs]
        imfs_spectra = np.split(merged_spectra, np.cumsum(imfs_len)[:-1])

        # 定义心率范围和相关参数
        heart_rate_range = (0.83, 2.5)
        min_hz, max_hz = heart_rate_range
        nyquist_freq = sample_rate / 2
        imfs_len = len(eemd_decomposition[0])
        min_freq_idx = int(min_hz / nyquist_freq * imfs_len)
        max_freq_idx = int(max_hz / nyquist_freq * imfs_len)

        # 初始化存储IMF的相关信息的数据结构
        imfs_info = [{'imf': imf,
                      'power_density_ratio': np.sum(spectrum[min_freq_idx:max_freq_idx]) / np.sum(spectrum),
                      'peak_ratio': spectrum[min_freq_idx:max_freq_idx].max() / spectrum.max()}
                     for imf, spectrum in zip(imfs, imfs_spectra)]

        # 排序IMFs，按照功率密度比和峰值比的组合权重排序
        imfs_info.sort(key=lambda x: (x['power_density_ratio'], x['peak_ratio']), reverse=True)

        # 选择综合得分最高的IMF作为心率信号
        final_imf = imfs_info[0]['imf']
        """
        return filted_signal

    def DisplaySignal(self):
        Sig_fore = np.array(self.processor.series_class.Sig_fore)
        Sig_left = np.array(self.processor.series_class.Sig_left)
        Sig_right = np.array(self.processor.series_class.Sig_right)
        if self.processor.series_class.Flag_Queue:
            if Sig_fore.size != 1:
                self.bvp_fore_raw = self.Mode(Sig_fore)
                self.quality_fore = 1 / \
                    (max(self.bvp_fore_raw)-min(self.bvp_fore_raw))
                self.bvp_fore = self.butterworth_filter(
                    self.processor.Signal_Preprocessing_single(self.bvp_fore_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                self.spc_fore = np.abs(np.fft.fft(self.bvp_fore))
                self.bpm_fore = self.processor.cal_bpm(
                    self.bpm_fore, self.spc_fore, self.processor.series_class.fps)
                if self.Data_ShowRaw:
                    self.Sig_f.setData(self.bvp_fore_raw, pen=(0, 255, 255))
                else:
                    self.Sig_f.setData(self.bvp_fore, pen=(0, 255, 255))
            else:
                self.Sig_f.setData([0], [0])
                self.Spec_f.setData([0], [0])
            if Sig_left.size != 1:
                self.bvp_left_raw = self.Mode(Sig_left)
                self.quality_left = 1 / \
                    (max(self.bvp_left_raw)-min(self.bvp_left_raw))
                self.bvp_left = self.butterworth_filter(
                    self.processor.Signal_Preprocessing_single(self.bvp_left_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                self.spc_left = np.abs(np.fft.fft(self.bvp_left))
                self.bpm_left = self.processor.cal_bpm(
                    self.bpm_left, self.spc_left, self.processor.series_class.fps)
                if self.Data_ShowRaw:
                    self.Sig_l.setData(self.bvp_left_raw, pen=(255, 0, 255))
                else:
                    self.Sig_l.setData(self.bvp_left, pen=(255, 0, 255))
            else:
                self.Sig_l.setData([0], [0])
                self.Spec_l.clear([0], [0])
            if Sig_right.size != 1:
                self.bvp_right_raw = self.Mode(Sig_right)
                self.quality_right = 1 / \
                    (max(self.bvp_right_raw)-min(self.bvp_right_raw))
                self.bvp_right = self.butterworth_filter(
                    self.processor.Signal_Preprocessing_single(self.bvp_right_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                self.spc_right = np.abs(np.fft.fft(self.bvp_right))
                self.bpm_right = self.processor.cal_bpm(
                    self.bpm_right, self.spc_right, self.processor.series_class.fps)
                if self.Data_ShowRaw:
                    self.Sig_r.setData(self.bvp_right_raw, pen=(255, 255, 0))
                else:
                    self.Sig_r.setData(self.bvp_right, pen=(255, 255, 0))
            else:
                self.Sig_r.setData([0], [0])
                self.Spec_r.setData([0], [0])
            self.quality_all = self.quality_fore+self.quality_left+self.quality_right
            self.confidence_fore = self.quality_fore/self.quality_all
            self.confidence_left = self.quality_left/self.quality_all
            self.confidence_right = self.quality_right/self.quality_all
            self.bpm_avg = self.bpm_fore*self.confidence_fore+self.bpm_left * \
                self.confidence_left+self.bpm_right*self.confidence_right
            #Label_Text = "Fs: \t\t{:.2f}\nFore BPM: \t{:.2f}\nLeft BPM: \t{:.2f}\nRight BPM:\t{:.2f}\n\nBPM Overall: \t{:.2f}".format(
            Label_Text = "Fs: \t\t{:.2f}\n\nBPM Overall: \t{:.2f}".format(
                self.processor.series_class.fps,
                #self.bpm_fore,
                #self.bpm_left,
                #self.bpm_right,
                self.bpm_avg
            )
            self.label.setText(Label_Text)
        else:
            self.Sig_f.setData([0], [0])
            self.Sig_l.setData([0], [0])
            self.Sig_r.setData([0], [0])
            self.label.setText(
                "Fs:\t\t"+str(self.processor.series_class.fps)+"\nData Collecting...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = mainwin()
    ui.show()
    sys.exit(app.exec_())