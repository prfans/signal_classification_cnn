#pragma once
#ifndef __H__INFERENCE_
#define __H__INFERENCE_

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

#define NORAML_SIZE_ALEX 124 // 归一化长度
#define NORAML_SIZE_DLA34 124 // 归一化长度
#define MAX_BATCH 16 //batch
#define MAX_NORMAL_SIZE 128 //归一化尺寸

//数据
typedef struct _tagData
{
	int size_; //数量
	float* data_; //数据地址
	int normal_size_; //数据归一化尺寸
}Data;

// 加载onnx模型
bool initial_onnx(const std::string modelPath, cv::dnn::Net& net);

// 释放
void release_onnx();

// 识别
int e13b_inference_b1(cv::dnn::Net& net, Data* data, int* recog_index, float* recog_prob);


// 归一化变量，类似numpy.resize
void numpy_normal(float* v, int n, float* new_v, int normal_n);


// 获取识别结果
void get_result(const cv::Mat& probBlob, std::vector<int>& output_sequence, std::vector<float>& output_prob);

// 数据初始化
void init_data(Data* data, float* data_buf, int normal_size);

//压入data、清零计数
bool push_to_data(Data* data, float* item, int item_length);
void clear_data(Data* data);

#endif
