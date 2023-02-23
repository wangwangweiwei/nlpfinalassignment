## 背景

本仓库是NLP期末作业代码仓库，代码有.py和.ipynb两种形式，代码内容是一样的。推荐下载.ipynb形式方便调试。

## 仓库结构

1. 文本分类。
   语料：清华大学相关实验室整理的新闻语料（前面所发）。总共14个类。
   要求：从每个类中，选择60-70%作为训练集，剩下的作为测试集，报告分类性能。
   分类模型：可以选择朴素贝叶斯、支持向量机、或者神经网络模型。

   

   nlp-4-1给出的代码基于[WOBERT](https://github.com/ZhuiyiTechnology/WoBERT)+Softmax实现。

   实验结果如下：

   <img src=".\img\nlp-4-1.png" alt="nlp-4-1" style="zoom:75%; float:left" />

2. 识别一个句子中的实体。
   具体要求和数据集请见：
   https://tianchi.aliyun.com/dataset/dataDetail?dataId=108771
   （前面所发的：dev.txt, test,txt, train.txt)
   在训练集上进行训练，在开发集上进行参数调整，在测试集上报告性能。
   用文本编辑软件打开时，选择UTF-8编码（NotePad++等工具都会自动识别的）。

   

   nlp-4-2给出的代码基于[BERT-BASE-CHINESE](https://huggingface.co/bert-base-chinese)+Softmax实现。

   实验结果如下：

   <img src=".\img\nlp-4-2.png" alt="nlp-4-2" style="zoom:75%; float:left" />

3.  文本语料相似度计算
   具体任务要求参考：https://zhuanlan.zhihu.com/p/51675979
   数据集：https://github.com/zejunwang1/CSTS

   nlp-4-2给出的代码基于[SBERT](https://www.sbert.net/)+Softmax实现。

   实验结果如下：

   <img src=".\img\nlp-4-3.png" alt="nlp-4-3" style="zoom:75%; float:left" />

