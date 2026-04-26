项目结构说明：
	evaluate中存放着评估的结果 指标为 AUC@5 AUC10 AUC@20  precision

	mnn_matching中是利用dino描述子进行mnn配对的结果。

	datases中存放着pairs文件以及用于test的数据集。navi测试集已经将最长边缩放为1024，scannet数据集将尺寸缩放为了【640*480】。如需进行训练需在官	网上下载完整数据集。navi数据集地址：https://github.com/google/navi.git。scannet数据集地址：https://github.com/ScanNet/ScanNet.git。
	测试用的pairs.txt文件格式为：每一行对应一个测试对，结构如下：path_image_A path_image_B exif_rotationA exif_rotationB [KA_0 ... KA_8] [KB_0 ... 	KB_8] [T_AB_0 ... T_AB_15]

	对dinov3模型进行微调后，只需按照mnn_matching中的csv格式得到两个数据集的matching文件夹。即可利用.\evaluate\evaluate_csv_essential.py进行指	标评估。


结果说明：
	AUC@5 AUC10 AUC@20  precision 用这四个指标进行了评估。AUC是在位姿估计领域比较常见的指标。precision是对于matching结果距离极线的距离进行的评估。具体含义可将评估脚本喂给ai进行讲解说明。
	按理说进行微调后这些指标都会相应的提高。

	可以认为是对matching任务进行的微调，也可认为是对pose estimation任务进行微调。


如果认为评价指标不饱满可以试着进行重定位实验或者matching真值坐标的评估。

以上是我直接利用dino通用描述符zero-shot（0修改）做的matching任务的全部内容。