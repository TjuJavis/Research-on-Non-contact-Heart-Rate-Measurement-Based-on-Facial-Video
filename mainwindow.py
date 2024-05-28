from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1180, 520)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.face = QtWidgets.QLabel(self.centralwidget)
        self.face.setGeometry(QtCore.QRect(380, 10, 780, 500))
        self.face.setText("")
        self.face.setObjectName("face")

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 200, 350, 300))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.Layout_BVP = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.Layout_BVP.setContentsMargins(0, 0, 0, 0)
        self.Layout_BVP.setObjectName("Layout_BVP")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)

        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 350, 180))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.Layout_button = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)

        self.Layout_button.setContentsMargins(0, 0, 0, 0)
        self.Layout_button.setObjectName("Layout_button")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_2)

        self.comboBox.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 28))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.Layout_button.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Button_RawTrue = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.Button_RawTrue.setObjectName("Button_RawTrue")
        self.horizontalLayout_2.addWidget(self.Button_RawTrue)
        self.Button_RawFalse = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.Button_RawFalse.setObjectName("Button_RawFalse")
        self.horizontalLayout_2.addWidget(self.Button_RawFalse)
        self.Layout_button.addLayout(self.horizontalLayout_2)

        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label.setMinimumSize(QtCore.QSize(0, 300))

        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.Layout_button.addWidget(self.label)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "GREEN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "GREEN-RED"))
        self.comboBox.setItemText(2, _translate("MainWindow", "CHROM"))
        self.comboBox.setItemText(3, _translate("MainWindow", "PBV"))
        self.Button_RawTrue.setText(_translate("MainWindow", "原始信号"))
        self.Button_RawFalse.setText(_translate("MainWindow", "滤波信号"))