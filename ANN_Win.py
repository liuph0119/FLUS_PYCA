#coding:utf-8
"""
作者：刘鹏华
版本：1.0
时间: 2017-12-20
模块功能：实现了一个继承自QWidget的窗体，实现了FLUS ANN-based Probability Occurrence Estimation的参数设置
"""


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ANNSettingWin(QWidget):

    def __init__(self):
        super().__init__()
        self.sWorkDir = "."
        self.initUI()

    def setWorkDir(self,wd):
        """
        设置当前项目的工作路径
        :param wd: 闯入的工作路径，形如"C:/data"
        :return:
        """
        self.sWorkDir = wd

    def initUI(self):

        # 加载土地利用文件
        QLabel("Landuse Path", self).move(15,10)
        self.LUPathLineEdit = QLineEdit("land use file name here...", self)
        self.LUPathLineEdit.setGeometry(100, 10, 300, 22)
        self.LUPathBtn = QPushButton("load", self)
        self.LUPathBtn.setGeometry(420, 10, 75, 22)
        self.LUPathBtn.clicked.connect(self.getOpenLUFileName)

        # 加载空间变量文件
        QLabel("Spatial Variables", self).move( 10, 50)
        self.FeatPathBtn = QPushButton("load", self)
        self.FeatPathBtn.setGeometry(420, 50, 75, 22)
        self.FeatPathBtn.clicked.connect(self.getOpenFeatureFileNames)
        self.FeatTextEdit = QTextEdit("Feature File Names Here...", self)
        self.FeatTextEdit.setGeometry(10, 80, 480, 100)

        # 采样方式，包括随机采样和均匀采样
        QLabel("Sampling", self).move(10, 190)
        self.sampleUniformRdbtn = QRadioButton("Uniform", self)
        self.sampleUniformRdbtn.setGeometry(10, 200, 100, 40)
        self.sampleUniformRdbtn.setChecked(True)
        self.sampleRandomRdbtn = QRadioButton("Random", self)
        self.sampleRandomRdbtn.setGeometry(150, 200, 100, 40)
        self.sampleRandomRdbtn.setChecked(False)

        # 采样比例，默认0.001
        QLabel("Rate",self).move(280, 210)
        self.sampleRatioSpinbox = QDoubleSpinBox(self)
        self.sampleRatioSpinbox.setGeometry(315, 210, 80, 20)
        self.sampleRatioSpinbox.setRange(0, 1)
        self.sampleRatioSpinbox.setSingleStep(0.001)
        self.sampleRatioSpinbox.setDecimals(3)
        self.sampleRatioSpinbox.setValue(0.001)

        # 神经网络隐含层数目，默认值8
        QLabel("Hidden Layer",self).move(10, 250)
        self.hiddenlayerSpinbox = QSpinBox(self)
        self.hiddenlayerSpinbox.setGeometry(100, 250, 50, 20)
        self.hiddenlayerSpinbox.setRange(2, 30)
        self.hiddenlayerSpinbox.setValue(8)

        # 神经网络最大迭代次数，默认值100
        QLabel("Max Iter", self).move(200, 250)
        self.maxIterSpinbox = QSpinBox(self)
        self.maxIterSpinbox.setGeometry(260, 250, 50, 20)
        self.maxIterSpinbox.setRange(1, 999)
        self.maxIterSpinbox.setValue(100)

        # 神经网络学习率（手动输入，默认值0.1）
        QLabel("Learning Rate", self).move(10, 280)
        self.lrLineEdit = QLineEdit("0.1",self)
        self.lrLineEdit.setGeometry(100, 280, 50, 20)

        # 两次迭代之间的最小误差之差（手动输入，默认值1e-6）
        QLabel("Alpha", self).move(200, 280)
        self.alphaLineEdit = QLineEdit("1e-6", self)
        self.alphaLineEdit.setGeometry(260, 280, 50, 20)

        QLabel("Save Path", self).move(10, 320)
        self.savePathLineEdit = QLineEdit("probability file path...", self)
        self.savePathLineEdit.setGeometry(100, 320, 300, 20)
        self.savePathBtn = QPushButton("save", self)
        self.savePathBtn.setGeometry(420, 320, 75, 22)
        self.savePathBtn.clicked.connect(self.getSaveProbabilityFileName)

        # 接收按钮，点击触发保存xml文件
        self.acceptBtn = QPushButton("accept", self)
        self.acceptBtn.setGeometry(200, 350, 100, 30)
        self.acceptBtn.clicked.connect(self.saveANNConfigureXML)


        self.setWindowTitle('ANN Settings')
        self.setWindowIcon(QIcon('./icons/earth_night.png'))
        self.setGeometry(300, 300, 500, 400)
        self.setStyleSheet('font-size: 10pt; font-family: Microsoft Yahei;')
        self.setFixedSize(500, 400)
        # self.show()



    @pyqtSlot()
    def getOpenLUFileName(self):
        """
        土地利用文件名称
        :return:
        """
        sFileName, ok1 = QFileDialog.getOpenFileName(self, "open land use file", self.sWorkDir, "TIFF(*.tif);;(*.*)")
        self.LUPathLineEdit.setText(sFileName)
        return sFileName

    @pyqtSlot()
    def getOpenFeatureFileNames(self):
        """
        批量选取空间变量文件
        :return: 文件名列表
        """
        vsFileNames, ok1 = QFileDialog.getOpenFileNames(self, "open feature files", self.sWorkDir, "TIFF(*.tif);;(*.*)")
        self.FeatTextEdit.clear()
        for sFileName in vsFileNames:
            self.FeatTextEdit.append("%s"%sFileName)
        return vsFileNames

    @pyqtSlot()
    def getSaveProbabilityFileName(self):
        sFileName, ok1 = QFileDialog.getSaveFileName(self, "open land use file", self.sWorkDir, "TIFF(*.tif);;(*.*)")
        self.savePathLineEdit.setText(sFileName)
        return sFileName

    @pyqtSlot()
    def saveANNConfigureXML(self):
        """
        将界面设置的参数存入xml文件
        :return:
        """
        from lxml import etree

        root = etree.Element("Configure")
        FilePathsElem = etree.SubElement(root, "FilePaths")
        SamplingElem = etree.SubElement(root, "Sampling")
        NNElem = etree.SubElement(root, "NeuralNetwork")

        etree.SubElement(FilePathsElem, "LandUseFilePath").text = self.LUPathLineEdit.text()
        etree.SubElement(FilePathsElem, "ProbabilityFilePath").text = self.savePathLineEdit.text()
        FeatureFilePathsElem = etree.SubElement(FilePathsElem, "FeatureFilePaths")
        fns = self.FeatTextEdit.toPlainText().split("\n")
        for i in range(len(fns)):
            etree.SubElement(FeatureFilePathsElem, "FeatureFilePath").text = fns[i]
        if(self.sampleRandomRdbtn.isChecked()):
            etree.SubElement(SamplingElem, "Method").text = "Random"
        else:
            etree.SubElement(SamplingElem, "Method").text = "Uniform"
        etree.SubElement(SamplingElem, "Ratio").text = str(self.sampleRatioSpinbox.value())
        etree.SubElement(NNElem, "HiddenLayer").text = str(self.hiddenlayerSpinbox.value())
        etree.SubElement(NNElem, "MaximumIteration").text = str(self.maxIterSpinbox.value())
        etree.SubElement(NNElem, "Alpha").text = str(self.alphaLineEdit.text())
        etree.SubElement(NNElem, "LearningRate").text = str(self.lrLineEdit.text())

        # 将参数设置写入xml文件
        doc = etree.ElementTree(root)
        doc.write(self.sWorkDir +"/ann_config.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')
        # 保存文件之后关闭窗口
        self.close()


