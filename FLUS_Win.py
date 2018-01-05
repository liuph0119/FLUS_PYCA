#!/usr/bin/python3
# -*- coding: utf-8 -*-
#function: QMainWindow cannot use layout，only QWidget

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import os, webbrowser, sys
from PIL import Image,  ImageQt
import FLUS_Utils, FLUS_Thread, FLUS_ANNSettingWin

# QGraphicsView
class GraphicsView(QGraphicsView):
    def __init__(self,parent=None, scene = None):
        super(GraphicsView, self).__init__(parent)
        self.scene = scene
        self.setScene(self.scene)


    def zoomFit(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def zoomIn(self):
        self.scale(1.1,1.1)

    def zoomOut(self):
        self.scale(0.9, 0.9)

    def wheelEvent(self, event):
        if event.angleDelta().y() / 120.0 > 0:
            factor = 1.1
        else:
            factor = 0.9
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        self.setDragMode(1)
        self.startPos = self.mapToScene(event.pos())


    def mouseReleaseEvent(self, event):
        pos = self.mapToScene(event.pos())
        dx = pos.x() - self.startPos.x()
        dy = pos.y() - self.startPos.y()

        rect = self.sceneRect().getRect()
        self.scene.setSceneRect(rect[0] - dx, rect[1] - dy, rect[2], rect[3])
        self.setDragMode(0)


class FLUS_Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWorkDir()

    def initUI(self):
        """
        初始化界面UI
        :return: None
        """
        # 绘图区
        self.scene = QGraphicsScene(self)
        self.view = GraphicsView(self, self.scene)
        #self.view.setScene(self.scene)
        self.view.setGeometry(100, 60, 395, 310)
        self.view.show()

        # 初始化菜单栏，工具栏及其对应的触发事件
        self.createActions()
        self.createMenuBar()
        self.createToolBar()

        # 设置窗口大小，位置，名称，logo
        self.setWindowTitle('FLUS')
        self.setWindowIcon(QIcon('./icons/earth_night.png'))
        self.setGeometry(300, 300, 500, 400)
        self.setFixedSize(500,400)

        self.show()

    def setWorkDir(self):
        """
        设置工作路径，所有临时文件从工作路径读取和写入
        :return:
        """
        self.sWorkDir = "."
        self.sWorkDir = QFileDialog.getExistingDirectory(self, "set work directory", ".", QFileDialog.ShowDirsOnly)
        self.statusBar().showMessage("Cur Dir: %s"%self.sWorkDir)

    # test function
    def display_image(self, fn):
        img = Image.open(fn)
        self.scene.clear()
        w, h = img.size
        self.imgQ = ImageQt.ImageQt(img)  # we need to hold reference to imgQ, or it will crash
        pixMap = QPixmap.fromImage(self.imgQ)
        self.scene.addPixmap(pixMap)
        self.view.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self.scene.update()
        self.view.repaint()
        self.view.show()

    def createActions(self):
        """
        创建事件/作用，及其连接的槽函数
        :return:
        """
        self.newProjAct = QAction(QIcon("icons/new.png"), "&New Project", self, triggered = self.setWorkDir)
        self.loadImgAct = QAction(QIcon("icons/open.png"), "&Load Image", self, triggered=self.addGeoTiffImage)
        self.clearAct = QAction(QIcon("icons/clear.png"), "&Clear", self, triggered=self.addGeoTiffImage)
        self.exitAct = QAction(QIcon("icons/shutdown.png"), "&Exit", self, triggered=qApp.quit)
        self.exitAct.setShortcut('Ctrl+Q')  # shortcut
        self.zoomInAct = QAction(QIcon("icons/zoom-in.png"), "&Zoom In", self, triggered=self.view.zoomIn)
        self.zoomOutAct = QAction(QIcon("icons/zoom-out.png"), "&Zoom Out", self, triggered=self.view.zoomOut)
        self.zoomFitAct = QAction(QIcon("icons/zoom-fit.png"), "&Zoom Fit", self, triggered=self.view.zoomFit)
        self.annAct = QAction(QIcon("icons/settings1.png"), "&ANN Settings", self, triggered=self.showANNSettingWin)
        self.runannAct = QAction(QIcon("icons/networking.png"), "&Run ANN", self, triggered = self.on_runANN)
        self.caAct = QAction(QIcon("icons/settings2.png"), "&CA Settings", self, triggered=self.addGeoTiffImage)
        self.runcaAct = QAction(QIcon("icons/start_here.png"), "&Run CA", self, triggered=self.on_runCA)
        self.aboutAct = QAction(QIcon("icons/about.png"), "&About", self, triggered=self.addGeoTiffImage)
        self.newversionAct = QAction(QIcon("icons/update.png"), "&New Version", self, triggered=self.gotoNewVersion)
        self.userguideenAct = QAction(QIcon("icons/pdf.png"), "&User Guide(en)", self, triggered=self.gotoUserGuideeEn)
        self.userguidechsAct = QAction(QIcon("icons/pdf.png"), "&User Guide(chs)", self, triggered=self.gotoUserGuideChs)
        self.precisionvalidationAct = QAction(QIcon("icons/ok.png"), "&Precision Validation", self, triggered=self.addGeoTiffImage)

    def createMenuBar(self):
        """
        创建菜单栏
        :return:  None
        """
        # QMenu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.newProjAct)
        self.fileMenu.addAction(self.loadImgAct)
        self.fileMenu.addAction(self.clearAct)
        self.fileMenu.addAction(self.exitAct)

        self.modelMenu = self.menuBar().addMenu("&FLUS Model")
        self.modelMenu.addAction(self.annAct)
        self.modelMenu.addAction(self.runannAct)
        self.modelMenu.addAction(self.caAct)
        self.modelMenu.addAction(self.runcaAct)

        self.validationMenu = self.menuBar().addMenu("&Validation")
        self.validationMenu.addAction(self.precisionvalidationAct)

        self.viewMenu = self.menuBar().addMenu("&View")
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.zoomFitAct)

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.newversionAct)
        self.helpMenu.addAction(self.userguideenAct)
        self.helpMenu.addAction(self.userguidechsAct)

    def createToolBar(self):
        """
        创建工具栏
        :return: None
        """
        # ToolBar
        self.toolbar = self.addToolBar('Load')
        self.toolbar.addAction(self.newProjAct)
        self.toolbar.addAction(self.loadImgAct)
        self.toolbar.addAction(self.clearAct)
        self.toolbar.addAction(self.exitAct)
        # self.toolbar.addAction()
        self.toolbar.addAction(self.zoomInAct)
        self.toolbar.addAction(self.zoomOutAct)
        self.toolbar.addAction(self.zoomFitAct)
        #
        #self.toolbar.addAction(self.annAct)
        self.toolbar.addAction(self.runannAct)
        self.toolbar.addAction(self.runcaAct)
        self.toolbar.addAction(self.aboutAct)

        # status bar
        self.statusBar().showMessage("FLUS ready")

        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(400, 400, 100, 0)
        self.progressBar.setHidden(True)

    # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # #
    # 以下为槽函数
    @pyqtSlot()
    def addGeoTiffImage(self):
        """
        加载Geo Tiff图像
        :return:
        """
        #QMessageBox.information(self,"Title",  "This is message!",   QMessageBox.Yes | QMessageBox.No)
        # get color
        #color = QColorDialog.getColor()
        sFileName, ok1 = QFileDialog.getOpenFileName(self, "open file", "./","TIFF(*.tif);;JPG(*.jpg);;PNG(*.png);;(*.*)")

        # 或者文件不存在
        if not os.path.exists(sFileName):
            self.statusBar().showMessage("File Does Not Exist!")
            return

        # 读取影像
        flusimg, msg = FLUS_Utils.loadImage(sFileName)
        # 读取失败
        if(flusimg == None):
            self.statusBar().showMessage("Load Fail! %s"%msg)
            return
        # 显示影像的基本信息
        vDataType = ["Unknown", "Byte", "UInt16", "Int16", "UInt32", "Int32", "Float32", "Float64", "CInt16", "CInt32", "CFloat32", "CFloat64"]
        self.statusBar().showMessage("Load Success! Row=%d Col=%d Band=%d Type=%s Nodata=%.0f"%(flusimg.mnRows, flusimg.mnCols, flusimg.mnBands,vDataType[flusimg.mgDataType],flusimg.mdInvalidValue))
        # QMessageBox.information(self, "load info", msg)
        del flusimg

    @pyqtSlot()
    def showANNSettingWin(self):
        # 显示计算发展概率的设置窗口
        self.annwin = FLUS_ANNSettingWin.ANNSettingWin()
        # 将当前工作区间传给设置窗口
        self.annwin.setWorkDir(self.sWorkDir)
        self.annwin.show()       # 需要通过self实例化为全局变量，不加self的话，一运行就被回收，也就无法显示。

    @pyqtSlot(str)
    def showMsg(self, msg):
        """
        状态栏显示日志，进度
        :param msg:
        :return:
        """
        self.statusBar().showMessage(msg)

    @pyqtSlot(str, str)
    def showMsgBox(self, msg, type):
        """
        显示不同类型的Message Box
        :param msg: 显示信息
        :param type: 对话框类型
        :return:
        """
        if(type == "info"):
            QMessageBox.information(self, "Information", msg)
        elif(type == "error"):
            QMessageBox.critical(self, "Error", msg)
        elif(type == "warn"):
            QMessageBox.warning(self, "Warning", msg)
        else:
            pass

    @pyqtSlot()
    def on_runANN(self):
        """
        创建线程，防止界面假死（未响应）
        :return:
        """
        loop = QEventLoop()

        trainingThread = FLUS_Thread.NNTrainingThread(self)
        trainingThread.setParam(self.sWorkDir, "ann_config.xml")
        trainingThread.start()

        # 线程信号与槽函数的连接
        trainingThread.sendMsg.connect(self.showMsg)
        trainingThread.sendMsg2Box.connect(self.showMsgBox)
        trainingThread.finished.connect(loop.quit)

        loop.exec_()
        trainingThread.wait()
        self.statusBar().showMessage("FLUS Ready @v@ ")

    @pyqtSlot()
    def on_runCA(self):
        loop = QEventLoop()
        caThread = FLUS_Thread.CAThread(self)
        caThread.setParam(self.sWorkDir, "ca_config.xml")
        caThread.start()

        caThread.sendMsg.connect(self.showMsg)
        caThread.sendMsg2Box.connect(self.showMsgBox)
        caThread.finished.connect(loop.quit)

        loop.exec_()
        caThread.wait()
        self.statusBar().showMessage("FLUS Ready @_@")

        pass

    @pyqtSlot()
    def gotoNewVersion(self):
        """
        访问flus主页
        :return:
        """
        webbrowser.open("http://www.geosimulation.cn/flus.html")

    @pyqtSlot()
    def gotoUserGuideeEn(self):
        """
        访问flus英文指南
        :return:
        """
        webbrowser.open("http://www.geosimulation.cn/FLUS/GeoSOS-FLUS%20Manual_En.pdf")

    @pyqtSlot()
    def gotoUserGuideChs(self):
        """
        访问flus中文指南
        :return:
        """
        webbrowser.open("http://www.geosimulation.cn/FLUS/GeoSOS-FLUS%20Manual_CHS.pdf")

    def __del__(self):
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FLUS_Win()
    sys.exit(app.exec_())




