import copy
from obspy.signal.detrend import polynomial
from scipy import signal
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from face2series import CAM2FACE
import numpy as np
import seaborn as sns
import time
sns.set()

class Series2rPPG:
    def __init__(self) -> None:
        # 从CAM加载历史序列
        self.series_class = CAM2FACE()
        self.Ongoing = True

    # 启动进程
    def PROCESS_start(self):
        try:
            self.series_class.PROCESS_start()
        except Exception as e:
            print(f"启动进程时发生错误: {e}")

    def _polynomial_detrend(self, sig, order=2):
        return polynomial(sig, order=order)

    def Signal_Preprocessing_single(self, sig):
        return self._polynomial_detrend(sig)

    def Signal_Preprocessing(self, rgbsig):
        data = np.array(rgbsig)
        data_r = self._polynomial_detrend(data[:, 0])
        data_g = self._polynomial_detrend(data[:, 1])
        data_b = self._polynomial_detrend(data[:, 2])

        return np.array([data_r, data_g, data_b]).T

    def PBV(self, signal):
        try:
            sig_mean = np.mean(signal, axis=1)

            sig_norm_r = signal[:, 0] / sig_mean[0]
            sig_norm_g = signal[:, 1] / sig_mean[1]
            sig_norm_b = signal[:, 2] / sig_mean[2]

            pbv_n = np.array(
                [np.std(sig_norm_r), np.std(sig_norm_g), np.std(sig_norm_b)])
            pbv_d = np.sqrt(
                np.var(sig_norm_r) + np.var(sig_norm_g) + np.var(sig_norm_b))
            pbv = pbv_n / pbv_d

            C = np.array([sig_norm_r, sig_norm_g, sig_norm_b])
            Ct = C.T
            Q = np.matmul(C, Ct)
            W = np.linalg.solve(Q, pbv)

            A = np.matmul(Ct, W)
            B = np.matmul(pbv.T, W)
            bvp = A / B
        except Exception as e:
            print(f"计算PBV时发生错误: {e}")
            bvp = np.array([])  # 发生错误时返回一个空数组

        return bvp

    def CHROM(self, signal):
        try:
            X = signal
            Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
            Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
            sX = np.std(Xcomp)
            sY = np.std(Ycomp)
            alpha = sX / sY
            bvp = Xcomp - alpha * Ycomp
        except Exception as e:
            print(f"计算CHROM时发生错误: {e}")
            bvp = np.array([])  # 发生错误时返回一个空数组

        return bvp

    # 优化后的PCA实现
    def PCA(self, signal):
        try:
            bvp = np.array([PCA(n_components=3).fit_transform(X)[:, 0] * PCA(n_components=3).fit_transform(X)[:, 1]
                            for X in signal])
        except Exception as e:
            print(f"计算PCA时发生错误: {e}")
            bvp = np.array([])  # 发生错误时返回一个空数组
        return bvp

    def GREEN(self, signal):
        return signal[:, 1]

    def GREEN_RED(self, signal):
        return signal[:, 1] - signal[:, 0]

    def cal_bpm(self, pre_bpm, spec, fps):
        return pre_bpm * 0.95 + np.argmax(spec[:int(len(spec) / 2)]) / len(spec) * fps * 60 * 0.05

    # 销毁对象
    def __del__(self):
        self.Ongoing = False

if __name__ == "__main__":
    processor = Series2rPPG()
    processor.PROCESS_start()