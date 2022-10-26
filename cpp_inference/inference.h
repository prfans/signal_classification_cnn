#pragma once
#ifndef __H__INFERENCE_
#define __H__INFERENCE_

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

#define NORAML_SIZE_ALEX 124 // ��һ������
#define NORAML_SIZE_DLA34 124 // ��һ������
#define MAX_BATCH 16 //batch
#define MAX_NORMAL_SIZE 128 //��һ���ߴ�

//����
typedef struct _tagData
{
	int size_; //����
	float* data_; //���ݵ�ַ
	int normal_size_; //���ݹ�һ���ߴ�
}Data;

// ����onnxģ��
bool initial_onnx(const std::string modelPath, cv::dnn::Net& net);

// �ͷ�
void release_onnx();

// ʶ��
int e13b_inference_b1(cv::dnn::Net& net, Data* data, int* recog_index, float* recog_prob);


// ��һ������������numpy.resize
void numpy_normal(float* v, int n, float* new_v, int normal_n);


// ��ȡʶ����
void get_result(const cv::Mat& probBlob, std::vector<int>& output_sequence, std::vector<float>& output_prob);

// ���ݳ�ʼ��
void init_data(Data* data, float* data_buf, int normal_size);

//ѹ��data���������
bool push_to_data(Data* data, float* item, int item_length);
void clear_data(Data* data);

#endif
