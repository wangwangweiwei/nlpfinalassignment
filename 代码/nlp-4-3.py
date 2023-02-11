#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    data_file = '/kaggle/working/sbert_text_similarity/data/atec_nlp_sim_train_all.csv'
    data = pd.read_csv(data_file, sep='\t', header=None, names=['index', 's1', 's2', 'label'])
    # 获取数据和标签
    x = data[['s1', 's2']].values.tolist()
    y = data['label'].values.tolist()
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
    print(x_train[0], y_train[0])
    print('总共有数据：{}条，其中正样本：{}条，负样本：{}条'.format(
        len(x), sum(y), len(x) - sum(y)))
    print('训练数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_train), sum(y_train), len(x_train) - sum(y_train)))
    print('测试数据：{}条,其中正样本：{}条，负样本：{}条'.format(
        len(x_test), sum(y_test), len(x_test) - sum(y_test)))
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = get_data()


# In[ ]:


pip install sentence_transformers


# In[ ]:


from torch.utils.data import DataLoader
import torch.nn as nn
from sentence_transformers import  SentenceTransformer, InputExample, losses
from sentence_transformers import models, evaluation
#from preprocess import get_data

model_path = '/kaggle/input/hflchineserobertawwmext/'
word_embedding_model = models.Transformer("hfl/chinese-roberta-wwm-ext", max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=256, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

x_train, x_test, y_train, y_test = get_data()
train_examples = []
for s, label in zip(x_train, y_train):
    s1, s2 = s
    train_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
test_examples = []
for s, label in zip(x_test, y_test):
    s1, s2 = s
    test_examples.append(
        InputExample(texts=[s1, s2], label=float(label))
    )
train_loader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.CosineSimilarityLoss(model)

model_save_path = '/kaggle/working/'
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
model.fit(train_objectives=[(train_loader, train_loss)],
          epochs=1,
          evaluator=evaluator,
          warmup_steps=100,
          save_best_model=True,
          output_path=model_save_path,)


# In[ ]:


import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util
#from preprocess import get_data

model_path = '/kaggle/working/'
model = SentenceTransformer(model_path)

#x_train, x_test, y_train, y_test = get_data()
s1 = np.array(x_test)[:, 0]
s2 = np.array(x_test)[:, 1]
embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)
pre_labels = [0] * len(s1)
predict_file = open('predict.txt', 'w')
for i in range(len(s1)):
    similarity = util.cos_sim(embedding1[i], embedding2[i])
    if similarity > 0.5:
        pre_labels[i] = 1
    predict_file.write(s1[i] + ' ' +
                       s2[i] + ' ' +
                       str(y_test[i]) + ' ' +
                       str(pre_labels[i]) + '\n')
print(classification_report(y_test, pre_labels))
predict_file.close()

