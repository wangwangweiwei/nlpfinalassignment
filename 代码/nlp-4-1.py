#!/usr/bin/env python
# coding: utf-8

# ### 生成实验数据

# In[ ]:


from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
import re
import os
import json


def count_text(category_path):
    """
    统计总体语料的分布情况
    :param category_path: 语料路径
    :return: 不同种类的语料字典
    """
    if os.path.exists(category_path):
        # 语料的路径
        category_path = category_path + '/*'  # 匹配所有的目录
        total = {}  # 语料总数
        for dir in glob(category_path):
            total[dir] = len(glob(dir + '/*.txt'))  # 每个类别下文件的数量
        print(total)
        print("文件共{}类,总的文件的数量:{}".format(len(total), sum(total.values())))
    else:
        raise FileExistsError('{}文件的路径不存在'.format(category_path))
    return total


def cut_corpus(path):
    """
    切分语料集 训练集 验证集 测试集，比例0.7:0.15:0.15
    :param path: 语料路径list
    :return: 切分后的数据集和标签
    """
    label = re.findall(r"[\u4e00-\u9fa5]+", path)  # 匹配汉字
    #label = path.split("/")[-1]
    files = glob(path + '/*.txt')  # 匹配txt文件的绝对路径
    # 切分数据集
    train, test = train_test_split(files, test_size=0.3, shuffle=True, random_state=2020)
    valid, test = train_test_split(test, test_size=0.5, shuffle=True, random_state=2021)
    print("train:{} test:{} valid:{}".format(len(train), len(test), len(valid)))
    return train, test, valid, label


def read_data(path, label=None, debug=False, frac=1):
    """
    读取文件中的数据title content
    :param path: 每条语料的路径信息list
    :param debug: 采样模式
    :param frac: 采样的比例
    :return:
    """
    titles = []
    contents = []
    for file in tqdm(path):
        with open(file, 'r', encoding='utf-8') as obj:
            data = obj.readlines()
        title = data[0].strip()
        content = [sen.strip() for sen in data[1:]]
        titles.append(title)
        contents.append(''.join(content))

    title_content = defaultdict(list)

    if len(titles) == len(contents):
        title_content['title'] = titles
        title_content['content'] = contents
        title_content['label'] = [label] * len(titles)
    else:
        raise ValueError('数据titles和contents数量不一致')
    df = pd.DataFrame(title_content, columns=['title', 'content', 'label'])
    if debug:
        # 采样
        df = df.sample(frac=frac, random_state=2020).reset_index(drop=True)
        print('采样的样本数量{}'.format(df.shape[0]))
    return df


def writ_to_csv(dictionary, filename='train'):
    """
    将数据写入csv文件
    :param dictionary: 字典格式
    :return:
    """
    df = pd.DataFrame(dictionary, columns=['title', 'content', 'label'])
    df.to_csv('{}.csv'.format(filename), sep='\t', index=False)
    print()
    print('writing succesfully')


def process(path, filename='train', frac=1):
    """
    读取数据文件将数据写入csv文件 title content label
    :param path: 数据文件的路径dict
    :param filename: 保存文件命名
    :return: None
    """
    print('loading {}'.format(filename))
    sample = []
    for label, data in path.items():
        under_sample = read_data(data, label, debug=True, frac=frac)
        sample.append(under_sample)
    df = pd.concat(sample, axis=0)
    print("{}文件的数据量为:{}".format(filename, df.shape[0]))
    # 保存文件的路径
    base_path = "/kaggle/working"
    save_path = base_path + '/' + filename + '.csv'
    df.to_csv(save_path, sep='\t', index=False)
    print('{} writing succesfully'.format(save_path))


def write_label_id(train_path,label_path):
    """标签映射为id"""
    data = pd.read_csv(train_path,header=0, delimiter="\t").dropna()
    label = data['label'].unique()
    print('标签:{}'.format(label))
    label2id = dict(zip(label, range(len(label))))
    json.dump(label2id, open(label_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    root_path = "/kaggle/working"
    category_path = "/kaggle/input/thucnews/THUCNews/THUCNews"
    # 语料的路径
    dir_dict = count_text(category_path)
    train_path = defaultdict(list)
    test_path = defaultdict(list)
    valid_path = defaultdict(list)
    for path in dir_dict.keys():
        # 切分数据集
        train, test, valid, label = cut_corpus(path)
        # 保存数据到字典
        train_path[label[0]] = train
        test_path[label[0]] = test
        valid_path[label[0]] = valid

    process(train_path, filename='train', frac=0.6)
    process(test_path, filename='test', frac=0.5)
    process(valid_path, filename='valid', frac=0.5)


    train_path = "/kaggle/working/train.csv"
    label_path = "/kaggle/working/label2id.json"
    # 生成标签到id的json文件
    write_label_id(train_path, label_path)


# ### 自己调用hugging face库实现

# In[ ]:


pip install rjieba


# In[1]:


import torch
from transformers import BertForMaskedLM as WoBertForMaskedLM
from transformers import RoFormerTokenizer as WoBertTokenizer
from transformers import AdamW


# In[ ]:


#考虑tokenizer放在collate_fn中还是放在dataset中啦，目前参考以前的例子放在前面
from torch.utils.data import Dataset
import pandas as pd
class BertDataset(Dataset):
    def __init__(self, path):
        super(BertDataset, self).__init__()
        self.data = pd.read_csv(path, header = 0, delimiter = "\t").dropna()
        #去除content列
        self.data.drop('content', axis = 1, inplace = True)
    def __getitem__(self, i):
        """
        title = self.data.loc[i]['title']
        label = self.data.loc[i]['label']
        """
        data = self.data.iloc[i]
        return data['title'], data['label']
    def __len__(self):
        return self.data.shape[0]


# In[3]:


path = "junnyu/wobert_chinese_base"
tokenizer = WoBertTokenizer.from_pretrained(path)
model = WoBertForMaskedLM.from_pretrained(path)


# In[ ]:


with open('/kaggle/input/download/Downloads/label2id.json', 'r', encoding='utf-8') as f:
    label2id = json.load(f)
def collate_fn(data):
    titles = [i[0] for i in data]
    labels = [i[1] for i in data]
    labels = [label2id.get(i) for i in labels]
    #编码
    data = tokenizer(titles,
                    truncation = True,
                    padding = True,
                    max_length = 128,
                    return_tensors = 'pt',
                    )
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置为0，其他位置是1
    input_ids = data['input_ids'].to(config.device)
    attention_mask = data['attention_mask'].to(config.device)
    token_type_ids = data['token_type_ids'].to(config.device)
    labels = torch.LongTensor(labels).to(config.device)
    
    return input_ids, attention_mask, token_type_ids, labels


# In[ ]:


from transformers import BertModel, BertConfig
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained('junnyu/wobert_chinese_base',num_labels = config.num_classes)
        self.bert = BertModel.from_pretrained("junnyu/wobert_chinese_base", config = model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        _, pooled = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids
                             ,return_dict=False)
        out = self.fc(pooled)
        return out


# In[ ]:


import time
import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (AdamW, BertTokenizer)




def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    print('User AdamW...')
    print(config.device)
    #初始化参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      eps=config.eps)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, mask, tokens, labels) in tqdm(enumerate(train_iter)):
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            outputs = model((trains, mask, tokens))
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            #             scheduler.step()
            if total_batch % 1000 == 0 and total_batch != 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                import torch
                predic = torch.max(outputs.data, 1)[1].cpu()
                
                train_acc = metrics.accuracy_score(true, predic)
                
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = round(time.time() - start_time, 4)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    
    #我认为训练的时候不能用训练集测试
    #test(config, model, test_iter)
    #显存资源比较紧缺，可以在每个epoch开始时释放下不用的显存资源
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config,
                                                                model,
                                                                test_iter,
                                                                test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = round(time.time() - start_time, 4)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, mask, tokens, labels in tqdm(data_iter):
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            
            outputs = model((texts, mask, tokens))
            loss = F.cross_entropy(outputs, labels)
            
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all,
                                               predict_all,
                                               target_names=config.label_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


# In[ ]:


import os
import json
import torch

class config:
    current_path = ""
    root_path = ""

    data_path = "/kaggle/input/thucnews/THUCNews/THUCNews"

    train_path = "/kaggle/input/download/Downloads/train.csv"
    test_path = "/kaggle/input/download/Downloads/test.csv"
    valid_path = "/kaggle/input/download/Downloads/valid.csv"

    label_path = "/kaggle/input/download/Downloads/label2id.json"


    is_cuda = True
    device = torch.device('cuda') if is_cuda else torch.device('cpu')

    """with open(root_path + '/data/stopwords.txt', "r", encoding='utf-8') as f:
        stopWords = [word.strip() for word in f.readlines()]
    """

    with open(label_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)

    label_list = label2id.keys()
    save_path = 'bert_wo_cls.pt'
    # bert
    eps = 1e-8
    learning_rate = 2e-5  # 学习率
    embedding_pretrained = None
    batch_size = 64
    hidden_size = 768
    num_epochs = 20
    dropout = 0.3  # 随机失活
    require_improvement = 1000 # 若超过1000batch效果还没提升，则提前结束训练
    num_classes = len(label2id)  # 类别数
    n_vocab = 50000  # 词表大小，在运行时赋值
    embed = 300


# In[32]:


#config = config()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样


print('Loading dataset')
train_dataset = BertDataset(config.train_path)
train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)
print('Loading dataset1')
dev_dataset = BertDataset(config.valid_path)
dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True)
'''print('Loading dataset2')
test_dataset = BertDataset(config.test_path, tokenizer=tokenizer, word=True)
test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn)'''
print('load network')
#model = Model(config).to(config.device)
    # 初始化参数

print('training model')
train(config, model, train_dataloader, dev_dataloader)
    # test(config, model, test_dataloader)  # 只测试模型的效


# In[33]:


print('Loading dataset2')
test_dataset = BertDataset(config.test_path)
test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn)
test(config, model, test_dataloader)  # 只测试模型的效果

