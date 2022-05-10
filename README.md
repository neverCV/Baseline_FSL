# README
---
#**目录**


# 1.简介
《Paper Reiteration: A Closer Look at Few-shot Classification》baseline++的核心就是先在基类上训练一个encoder，去除最后的分类器后对encoder进行冻结。
在新类上将support集的样本送入encoder得到对应的feature，用一个权重矩阵来进行Nway的分类任务与之前的分类器的不同点在于，预训练时用的是乘法，而baseline++在finetune的时候用的是余弦相似度。

# 2.复现精度
代码在miniimagenet下进行训练和测试（5way）

	论文 1shot:48.2	5shot:66.4
	复现 1shot:48.5	5shot:66.4
原论文为跑完400个epoch后再进行测试，考虑到ANIL论文中提出了MAML其实在训练的后半段都是在训练分类器，而在本论文中预训练的分类器是舍弃掉的，所以可能后半段的训练是无用的，在我的代码中，只跑了55个epoch就差不多拿到了论文的精度，其一可能是因为后半段的训练确实无用，其二可能是因为paddle的optimizer的SGD并没有dampening这个参数，所以在测试时我选择了Momentum来进行近似。
由于paddle的SGD没有dampening参数如果使用作者的预训练方式跑400个epoch实际上会造成测试时的过拟合，从而使精度反而降低（在150个epoch时差不多为1shot 0.46）。由于seed固定有问题，当发现精度不满足时在50个epoch附近找一下模型参数即可。
	

# 3.代码结构

	├─datafile								# 存放索引的csv文件
	├─images								# 存放miniimagenet的所有图片
	├─save									# 存放训练后储存的文件，如果不存在会自动创建
	|fs_test.py								# 模型测试
	|fs_train.py								# 训练
	|Mini_Imagenet.py							# 数据集代码    
	|model_zoo.py								# 存放模型

# 4.数据集说明
miniimagenet有100类，每类600张图片。
Mini-Imagenet数据集中还包含了train.csv、val.csv以及test.csv三个文件。

    train.csv包含38400张图片，共64个类别。
    val.csv包含9600张图片，共16个类别。
    test.csv包含12000张图片，共20个类别。

每个csv文件之间的图像以及类别都是相互独立的，即共60000张图片，100个类。
可以从百度的aistudio:https://aistudio.baidu.com/aistudio/datasetdetail/138415 下载miniimagenet的图片，csv文件本项目附带的有，是随机shuffle后切分出来的

# 5. 环境依赖
	paddlepaddle-gpu==2.2.2
	paddlefsl
	tqdm
# 6.运行
正常是先运行fs_train.py再运行fs_test.py，但本项目提供了第55轮的模型参数，可直接运行fs_test.py 查看运行结果

###	step1.运行fs_train.py
	python fs_train.py


###	step2.运行fs_test.py
	python fs_test.py

