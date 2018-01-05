#!/usr/bin/python3
# -*- coding: utf-8 -*-
from osgeo import gdal
import osr
from gdalconst import *
import numpy as np
import os

# tif影像类，包含必备参数
class FLUSImage:
    mpoDataset = None

    mnRows = 0
    mnCols = 0
    mnBands = 0
    mgDataType = GDT_Byte
    mpGeoTransform = None
    msProjectionRef = None
    mdInvalidValue = 0

    mpArray = None






def loadImage(fn):
    """
    读取geo-tiff影像
    :param fn: 文件名
    :return: FLUSImage对象，读取是否成功的信息
    """
    # check if file exists
    if not os.path.exists(fn):
        return None,"%s does not exist!"%fn

    flusimg = FLUSImage()
    try:
        flusimg.mpoDataset = gdal.Open(fn, GA_ReadOnly)
        flusimg.mnRows = flusimg.mpoDataset.RasterYSize
        flusimg.mnCols = flusimg.mpoDataset.RasterXSize
        flusimg.mnBands = flusimg.mpoDataset.RasterCount
        flusimg.mgDataType = flusimg.mpoDataset.GetRasterBand(1).DataType
        flusimg.mdInvalidValue = flusimg.mpoDataset.GetRasterBand(1).GetNoDataValue()

        flusimg.mpArray = np.zeros((flusimg.mnBands, flusimg.mnRows, flusimg.mnCols))
        for i in range(flusimg.mnBands):
            data = flusimg.mpoDataset.GetRasterBand(i + 1).ReadAsArray(0, 0, flusimg.mnCols, flusimg.mnRows)
            flusimg.mpArray[i] = data

        # msg = "Image Information:\n  row=" + str(flusimg.mnRows) + "\n  col=" + str(flusimg.mnCols) + "\n  band=" + \
        #       str(flusimg.mnBands) + "\n  datatype=" + str(flusimg.mgDataType) + "\n  nodata=" + str(
        #     flusimg.mdInvalidValue) + "\n"

        return flusimg, "open success"
    except:
        return None, "open error!"


def randomSample2table(featuresArr, landuseArr, r = 0.001,  lu_nodata = 255, feat_nodata = None):
    """
    从空间变量和土地利用类型中随机采集样本
    整体思路：
        1. 确保影像的行列数目一致
        2. 土地利用数据为单波段的三维矩阵，特征数据集为组合的多波段的三维矩阵
        3. 根据采样比例，像元数目确定所需采集的样本数目
        3. 将所有位置（行数*列数）的索引随机打乱顺序，从中从头至尾选取一个位置的数据，分别与其对应的nodata对比，如果数据均有效，且样本数目未达到需求， 则存入二维数组
    :param featuresArr: 空间变量3维numpy矩阵，分别为层、行数目，列数目
    :param landuseArr: 土地利用数据3维numpy矩阵，但只有一层
    :param r: 采样比例
    :param feat_nodata: 特征数据集的nodata值, list
    :param lu_nodata: 土地利用数据的nodata值, double
    :return: 样本二维numpy矩阵[[x1,x2,...xn,y]...]
    """
    # 特征集和土地利用数据的行列数目不一致，空间变量数目与其nodata不匹配
    if(featuresArr.shape[1] != landuseArr.shape[1] or
    featuresArr.shape[2] != landuseArr.shape[2] or
    featuresArr.shape[0] != len(feat_nodata)):
        return None


    # 行列索引打乱顺序
    nSumPixelNum = landuseArr.shape[1]*landuseArr.shape[2]
    vIndexs = np.array([i for i in range(nSumPixelNum)])
    np.random.shuffle(vIndexs)

    # 计算采样数目
    num_sample = int(nSumPixelNum*r)
    samples = np.zeros((num_sample, len(featuresArr) + 1))
    num = 0     # 实际采集的样本数目
    times = 0   # 迭代次数
    # 循环，终止条件是样本数目达到了需求，或者所有像元均被遍历完
    while num < num_sample and times < nSumPixelNum:
        # 从随机列表头中拿出一个下标（行列位置）
        _i = vIndexs[times]
        _col = int(_i % (landuseArr.shape[1]))
        _row = int((_i - _col) / (landuseArr.shape[2]))

        # 如果该位置的土地利用类型或者空间变量值为 nodata，则跳过
        # 采用“与运算”，即任意图层为nodata都跳过
        nodataflag = True and abs(landuseArr[0, _row, _col] - lu_nodata) > 1e-6
        for i in range(featuresArr.shape[0]):
            nodataflag = nodataflag and abs(featuresArr[i, _row, _col] - feat_nodata[i]) > 1e-6

        if(nodataflag == False):
            times += 1
            continue

        # 将各类特征和土地利用类型存入样本表格
        for k in range(featuresArr.shape[0]):
            samples[num, k] = featuresArr[k, _row, _col]
        samples[num, len(featuresArr)] = landuseArr[0, _row, _col]
        num += 1
        times += 1
    # 返回实际有效的样本数目
    del vIndexs
    return samples[:num, :]


def uniformSample2table(featuresArr, landuseArr, r = 0.001,  lu_nodata = 255, feat_nodata = None):
    """
    从空间变量和土地利用类型中均匀采集样本
    整体思路：
        1. 确保影像的行列数目一致
        2. 土地利用数据为单波段的三维矩阵，特征数据集为组合的多波段的三维矩阵
        3. 根据采样比例，像元数目，类别数目确定各类所需采集的样本数目
        4. 将所有位置（行数*列数）的索引随机打乱顺序，从中从头至尾选取一个位置的数据，分别与其对应的nodata对比，如果数据均有效，且该类样本数目未达到需求，则存入二维数组
    :param featuresArr: 空间变量3维numpy矩阵，分别为层、行数目，列数目
    :param landuseArr: 土地利用数据3维numpy矩阵，但只有一层
    :param r: 采样比例
    :param feat_nodata: 特征数据集的nodata值, list
    :param lu_nodata: 土地利用数据的nodata值, double
    :return: 样本二维numpy矩阵[[x1,x2,...xn,y]...]
    """
    # 特征集和土地利用数据的行列数目不一致，空间变量数目与其nodata不匹配
    if(featuresArr.shape[1] != landuseArr.shape[1] or
    featuresArr.shape[2] != landuseArr.shape[2] or
    featuresArr.shape[0] != len(feat_nodata)):
        return None


    # 行列索引打乱顺序
    nSumPixelNum = landuseArr.shape[1]*landuseArr.shape[2]
    vIndexs = np.array([i for i in range(nSumPixelNum)])
    np.random.shuffle(vIndexs)

    # 类别有效值（包含了nodata）
    validVals = np.unique(landuseArr)
    classNum = validVals.shape[0]
    if(lu_nodata in validVals):
        classNum = validVals.shape[0]-1


    # 计算采样数目
    num_sample = int(nSumPixelNum*r)
    # 每一类需要采集的样本数目
    num_sample_each_category = [int(num_sample/classNum) for i in range(classNum)]
    num_sample_each_category_actual = [0 for i in range(classNum)]


    samples = np.zeros((num_sample, len(featuresArr) + 1))
    num = 0
    times = 0
    # 循环，终止条件是样本数目达到了需求，或者所有像元均被遍历完
    while num < num_sample and times < nSumPixelNum:
        # 从随机列表头中拿出一个下标（行列位置）
        _i = vIndexs[times]
        _col = int(_i % (landuseArr.shape[1]))
        _row = int((_i - _col) / (landuseArr.shape[2]))

        # 如果该位置的土地利用类型或者空间变量值为 nodata，则跳过
        nodataflag = True and abs(landuseArr[0, _row, _col] - lu_nodata) > 1e-6
        for i in range(featuresArr.shape[0]):
            nodataflag = nodataflag and abs(featuresArr[i, _row, _col] - feat_nodata[i]) > 1e-6

        if(nodataflag == False):
            times += 1
            continue

        # 如果该类所需的样本数目达到了，则跳过
        # 土地利用类型从1开始，因此下标需要减1
        if (num_sample_each_category_actual[int(landuseArr[0, _row, _col]-1)] >= num_sample_each_category[
            int(landuseArr[0, _row, _col]-1)]):
            times += 1
            continue

        # 将各类特征和土地利用类型存入样本表格
        for k in range(featuresArr.shape[0]):
            samples[num, k] = featuresArr[k, _row, _col]
        samples[num, len(featuresArr)] = landuseArr[0, _row, _col]
        num += 1
        times += 1
        num_sample_each_category_actual[int(landuseArr[0, _row, _col]-1)] += 1
    # 返回实际有效的样本数目
    del vIndexs
    return samples[:num, :]


def array2Tif(arr, outputfn, datatype = gdal.GDT_Float32, Nodata = None, dpGeoTransform = None, sSpatialProj = None):
    """
    存储为 geo-tif 文件
    :param arr: 数组，三维
    :param outputfn: 输出文件名
    :param datatype: 数据类型
    :param Nodata: nodata
    :param dpGeoTransform: 地理转换参数，由6个double数组成的list
    :param sSpatialProj: 空间投影, wkt
    :return:
    """
    (nBands, nRows, nCols) = arr.shape
    driver = gdal.GetDriverByName('GTiff')
    # 文件存在则退出
    if os.path.exists(outputfn):
        try:
            os.remove(outputfn)
        except:
            return None


    outRaster = driver.Create(outputfn, nCols, nRows, nBands, datatype)
    # 设置地理转换参数，投影信息
    outRaster.SetGeoTransform(dpGeoTransform)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(sSpatialProj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

    # 分别存储每一个波段
    for band_num in range(1, nBands + 1):
        outband = outRaster.GetRasterBand(band_num)
        # 设置nodata值，写入数组
        outband.SetNoDataValue(Nodata)
        outband.WriteArray(arr[band_num - 1, :, :])
        outband.FlushCache()
    return 1


def rouletteWheelSelection(arr):
    """
    轮盘赌选择，返回index和value
    :param arr: numpy 1维数组, 形如[0.1, 0.3, 0.6]
    :return: index 和 value
    """
    total = arr.sum()
    r = total * np.random.uniform()

    tmp = 0
    for i in range(arr.shape[0]):
        tmp += arr[i]
        if(tmp > r):
            return i, arr[i]



