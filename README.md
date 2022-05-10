# README
---
#**目录**


# 1.简介
《Paper Reiteration: A Closer Look at Few-shot Classification》baseline++的核心就是先在基类上训练一个encoder，去除最后的分类器后对encoder进行冻结。
在新类上将support集的样本送入encoder得到对应的feature，用一个权重矩阵来进行Nway的分类任务与之前的分类器的不同点在于，预训练时用的是乘法，而baseline++在finetune的时候用的是余弦相似度。

# 2.复现精度
代码在miniimagenet下进行训练和测试（5way）

	论文 1shot:48.2	5shot:66.4
	复现 1shot:48.5	5shot:66.6
	

# 3.代码结构

	├─datafile								# 存放索引的csv文件
	├─images								# 存放miniimagenet的所有图片
	|fs_test.py								# 模型测试
	|fs_train.py							# 训练
	|Mini_Imagenet.py						# 数据集代码    
	|model_zoo.py							# 存放模型

# 4.数据集说明
miniimagenet有100类，每类600张图片。
Mini-Imagenet数据集中还包含了train.csv、val.csv以及test.csv三个文件。

    train.csv包含38400张图片，共64个类别。
    val.csv包含9600张图片，共16个类别。
    test.csv包含12000张图片，共20个类别。

每个csv文件之间的图像以及类别都是相互独立的，即共60000张图片，100个类。

# 5. 环境依赖
	paddlepaddle-gpu==2.2.2
	paddlefsl
	tqdm
# 6.运行

###	step1.运行fs_train.py
	python fs_train.py


###	step2.运行fs_test.py
	python fs_test.py

