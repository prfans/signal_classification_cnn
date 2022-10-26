
#include "inference.h"


//dnn加载PYTORCH模型文件
bool initial_onnx(const std::string modelPath, cv::dnn::Net& net)
{
	std::string dat_file = modelPath;

	net = cv::dnn::readNetFromONNX(dat_file);
	if (net.empty())
		return false;

	// computation backends
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);

	// computations on specific target device
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	return true;
}

void release_onnx()
{
	//
}

void init_data(Data* data, float* data_buf, int normal_size)
{
	data->size_ = 0;
	data->data_ = data_buf;
	data->normal_size_ = normal_size;
}

bool push_to_data(Data *data, float* item, int item_length)
{
	if (data->size_ >= MAX_BATCH)
		return false;

	numpy_normal(item, item_length, data->data_ + data->size_ * data->normal_size_, data->normal_size_);
	data->size_++;

	return true;
}

void clear_data(Data* data)
{
	data->size_ = 0;
}


int e13b_inference_b1(cv::dnn::Net& net, Data* data, int* recog_index, float* recog_prob)
{
	if (data->size_ != 1)
	{
		return -1;
	}

	// 生成张量bx1x12x
	const int sizes[] = { 1, 1, data->normal_size_ };
	cv::Mat inputBlob(3, sizes, CV_32F, data->data_);

	// 前向传播
	net.setInput(inputBlob, "data");
	cv::Mat pred = net.forward("output");

	// 识别结果解析
	std::vector<int> output_sequence;
	std::vector<float> output_prob;
	get_result(pred, output_sequence, output_prob);

	for (int k = 0; k < data->size_; k++)
	{
		recog_index[k] = output_sequence[k];
		recog_prob[k] = output_prob[k];
	}

	return 0;
}

void get_result(const cv::Mat& probBlob, std::vector<int>& output_sequence, std::vector<float>& output_prob)
{
	int nblock_num = 0;//划分block个数；
	int pro_num = 0;//标签总数
	int ndim = 0;//概率矩阵维数
	int blank_index_ = -1; //默认空格 label=0

	nblock_num = probBlob.size[0];
	cv::Mat outBatch = probBlob.reshape(1, nblock_num);
	pro_num = outBatch.size[1];
	int prev_class_idx = -1;

	//每个小窗口下 各个标签的概率值，取最大值
	for (int n = 0; n < nblock_num; n++)
	{
		float sum_probs = 0.0f;
		int max_class_idx = 0;
		float max_prob = outBatch.ptr<float>(n)[0];
		for (int t = 0; t < pro_num; t++)
		{
			float prob = outBatch.ptr<float>(n)[t];
			if (prob > max_prob)
			{
				max_prob = prob;
				max_class_idx = t;
			}

			// sum_probs += exp(prob - max_prob)
			sum_probs += exp(prob); //softmax
		}
		if (max_class_idx != blank_index_)
		{
			output_sequence.push_back(max_class_idx);
			output_prob.push_back(exp(max_prob) / (sum_probs + FLT_MIN));
		}

		prev_class_idx = max_class_idx;
	}
}

//信号归一化操作，与pytorch对应
void numpy_normal(float* v, int n, float* new_v, int normal_n)
{
	float max_v = FLT_MIN, min_v = FLT_MAX;

	// 最大值 最小值
	for (int k = 0; k < n; k++)
	{
		if (max_v < v[k])
			max_v = v[k];
		if (min_v > v[k])
			min_v = v[k];
	}

	// 归一化
	float normal_value = (max_v - min_v + FLT_MIN);

	int k;
	int length = std::min(normal_n, n);
	for (k = 0; k < length; k++)
	{
		new_v[k] = 2 * (v[k] - min_v) / normal_value - 1.0;
	}
	for (; k < normal_n; k++)
	{
		new_v[k] = 0.0f;
	}
}

