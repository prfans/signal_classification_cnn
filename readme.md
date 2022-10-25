
## 一维信号分类训练及测试
    支持alexnet、dlanet、mobilenetv2、resnet、senet、shufflenetv2、squeezenet等多种骨干网
    
## 训练自己的数据
    数据格式为txt格式，数据格式见data/example.txt，每个信号存为一个txt文件。
    数据存放方式：每个类别按照不同文件夹存放，例如共3类，0文件夹存放0类的数据，1文件夹存放类型1的数据，2文件夹存放类型2的数据数据存放方式：每个类别按照不同文件夹存放，例如共3类，0文件夹存放0类的数据，1文件夹存放类型1的数据，2文件夹存放类型2的数据.....
    在tran.py更改训练训练数据与测试数据路径，然后执行python train.py进行训练。

## 测试
    python detect.py
    
## 导出onnx格式，部署可使用opencv
    python export.py
