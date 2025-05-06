import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制禁用所有 GPU，仅使用 CPU 运算

import random
import re
import jieba
import warnings
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Layer
import tensorflow as tf
import ast
import csv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

#######################3
# part-0: 调整超参数
######################
my_lr = 1e-2
my_test_size = 0.1
my_validation_split = 0.1
my_epochs = 40
my_batch_size = 128
my_dropout = 0.2

my_optimizer = Nadam(learning_rate=my_lr)

my_loss = 'binary_crossentropy'


###################
# part-A: 数据探索
####################
print('\npart-A: 数据探索')


# 将所有的评论内容放置到一个list里，列表中的每个元素是一条评论
train_texts_orig = [] 

# 读取正面评论、负面评论各2000条至一个list中
with open('./data/positive_samples.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data = ast.literal_eval(line)
            train_texts_orig.append(data['text'])

with open('./data/negative_samples.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data = ast.literal_eval(line)
            train_texts_orig.append(data['text'])

########################
#part-B: 数据预处理-分词
######################
print('\npart-B: 数据预处理-分词')

# 使用gensim加载已经训练好的汉语词向量
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram.bz2', binary=False)

# 用jieba进行中文分词，最后将每条评论转换为了词索引的列表
train_tokens = []
for text in train_texts_orig:
    # 分词前去掉标点和特殊符号
    text = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            # 将分出来的每个词转换为词向量中的对应索引
            cut_list[i] = cn_model.key_to_index[word]
        except KeyError:
            # 如果词不在词向量中，则索引标记为0
            cut_list[i] = 0
    train_tokens.append(cut_list)
print('num of train_tokens: {0}'.format(len(train_tokens))) # 4000

#########################
#part-C: 数据预处理-索引化
#########################
print('\npart-C: 数据预处理-索引化')

# 获得每条评论的长度，即分词后词语的个数，并将列表转换为ndarray格式
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)

print('max-len of train_tokens: {0}'.format(np.max(num_tokens)))  # 最长评价的长度 1438
print('mean-len of train_tokens: {0}'.format(np.mean(num_tokens)))  # 平均评论的长度 68.77625

# 绘制评论长度直方图
plt.hist(np.log(num_tokens), bins = 100)
plt.xlim((0,10))
plt.ylabel('num of train_tokens')
plt.xlabel('len of train_tokens')
plt.show()

# 每段评语的长度不一，需要将索引长度标准化
mid_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
mid_tokens = int(mid_tokens)
rate = np.sum( num_tokens < mid_tokens ) / len(num_tokens)
print('selected mid-len of train_tokens: {0}'.format(mid_tokens)) # 选取一个平均值，尽可能多的覆盖
print('cover rate: {0}'.format(rate)) # 覆盖率


##################################
#part-D: 数据预处理-重新构建词向量
####################################
print('\npart-D: 数据预处理-重新构建词向量')

# print('num of vector: {0}'.format(len(cut_list))) # 预训练的词向量词汇数

# 为了节省训练时间，抽取前50000个词构建新的词向量
num_words = 50000
embedding_dim = 300

# 初始化embedding_matrix，之后在keras上进行应用
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵，维度为 50000 * 300
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')



# 新建词向量的维度，keras会用到
embedding_matrix.shape # (50000, 300)

####################################
#part-E: 数据预处理-填充与裁剪
###################################
print('\npart-E: 数据预处理-填充与裁剪')

# 输入的train_tokens是一个list，返回的train_pad是一个numpy array，采用pre填充的方式
train_pad = pad_sequences(train_tokens, maxlen=mid_tokens, padding='pre', truncating='pre')

# 超出五万个词向量的词用0代替
train_pad[train_pad>=num_words] = 0

# 准备实际输出结果向量向量，前2000好评的样本设为1，后2000差评样本设为0
train_target = np.concatenate((np.ones(2000),np.zeros(2000)))  #加标签

#####################
#part-F: 训练
#####################
print('\npart-F: 训练')

# 用sklearn分割训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=my_test_size, random_state=12)

# 搭建神经网络
model = Sequential()

model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=mid_tokens, trainable=False))
model.add(Bidirectional(LSTM(units=32, dropout=my_dropout, return_sequences=True)))
model.add(LSTM(units=16, dropout=my_dropout, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=my_loss, optimizer=my_optimizer, metrics=['accuracy'])

# # 查看模型的结构
model.summary()


###########################
# part-G: 调试
##########################
print('\npart-G: 调试')

# 建立一个权重的存储点，保存训练中的最好模型
path_checkpoint = 'tmp\\weights.weights.h5'
checkpointer = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss' , verbose=1 , save_weights_only=True , save_best_only=True)


# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)

# 定义callback函数
callbacks = [earlystopping, checkpointer, lr_reduction]

# 开始训练
history = model.fit(X_train, y_train, validation_split=my_validation_split, epochs=my_epochs, batch_size=my_batch_size, callbacks=callbacks)


# 模型可视化-历史
plt.figure(figsize=(11, 4))

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'validation'])

plt.figure(1)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'validation'])

plt.show()

# 模型可视化-RNN
plot_model(model, show_shapes=True, show_layer_names=True, to_file='tmp\\model.png')

# 模型评估-准确率
result = model.evaluate(X_test, y_test, verbose=0)
print('Loss: {0:.4}'.format(result[0]))
print('Accuracy: {0:.4%}'.format(result[1]))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 获取模型预测的概率
y_probs = model.predict(X_test).flatten()

# 计算 ROC 曲线数据
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 选择最优阈值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f'\n=== ROC分析结果 ===')
print(f'AUC值：{roc_auc:.4f}')
print(f'推荐的最佳分类阈值：{optimal_threshold:.4f}')

# 绘制 ROC 曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # 参考线
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Best Threshold: {optimal_threshold:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

import jieba
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense

# ======================
# 1. 加载训练好的模型和预处理组件
# ======================
# 预处理参数
num_words = 50000
embedding_dim = 300
mid_tokens = 680
max_len = mid_tokens
my_dropout = 0.2

# 加载词向量
from gensim.models import KeyedVectors
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram.bz2', binary=False)

# 构建 embedding_matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')

# 先重建模型结构
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=mid_tokens, trainable=False))
model.add(Bidirectional(LSTM(units=32, dropout=my_dropout, return_sequences=True)))
model.add(LSTM(units=16, dropout=my_dropout, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# 手动 build，设置输入形状
model.build(input_shape=(None, mid_tokens))

model.load_weights('tmp/weights.weights.h5')  

cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram.bz2', binary=False)


# ======================
# 2. 定义预处理函数
# ======================
def preprocess_new_text(text):
    """对新评论进行与训练数据一致的预处理"""
    # 1. 清洗标点符号（与训练时完全相同）
    text = re.sub(r"[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", text)
    
    # 2. 分词处理
    cut_list = list(jieba.cut(text))
    
    # 3. 转换为词索引（处理OOV + 超限）
    indexed_tokens = []
    for word in cut_list:
        try:
            idx = cn_model.key_to_index[word] + 1
            if idx >= num_words:
                idx = 0
        except KeyError:
            idx = 0
        indexed_tokens.append(idx)

    
    # 4. 填充/截断序列
    padded = pad_sequences([indexed_tokens], maxlen=max_len, padding='pre', truncating='pre')
    return padded[0]

# ======================
# 3. 数据加载与解析
# ======================


def load_new_dataset(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for parts in reader:
            if len(parts) >= 2:
                try:
                    label = int(parts[0])
                    review = parts[1].replace('""', '"')
                    samples.append((label, review))
                except Exception as e:
                    print(f"跳过异常行: {parts}, 错误: {e}")
    return samples


# ======================
# 4. 执行预测并保存结果
# ======================
def predict_and_save(samples, positive_file, negative_file):
    """预测并保存结果到文件"""
    with open(positive_file, 'w', encoding='utf-8') as pos_f, \
         open(negative_file, 'w', encoding='utf-8') as neg_f:

        for label, review in samples:
            # 预处理文本
            processed = preprocess_new_text(review)
            
            # 进行预测（添加batch维度）
            prediction = model.predict(np.array([processed]), verbose=0)[0][0]
            
            # 转换预测结果为0/1（阈值设为最佳阈值optimal_threshold）
            final_label = 1 if prediction >= optimal_threshold else 0
            
            # 构建字典字符串（保持与示例一致的单引号格式）
            # 构建字典字符串（保持与示例一致的单引号格式）
            safe_review = review.replace("'", "''")
            result = f"{{'text':'{safe_review}','label':{final_label}}}\n"

            
            # 写入对应文件
            if final_label == 1:
                pos_f.write(result)
            else:
                neg_f.write(result)

# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    # 加载新数据
    new_samples = load_new_dataset('./data/ChnSentiCorp_htl_all.csv')
    
    # 执行预测并保存结果
    predict_and_save(
        new_samples,
        positive_file='./predict/predicted_positive.txt',
        negative_file='./predict/predicted_negative.txt'
    )
    print("预测完成！结果已保存到 predicted_positive.txt 和 predicted_negative.txt")
    # 可视化：预测概率直方图
    print("正在绘制预测概率分布图...")
    probs = []
    for _, review in new_samples:
        processed = preprocess_new_text(review)
        prediction = model.predict(np.array([processed]), verbose=0)[0][0]
        probs.append(prediction)


    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.hist(probs, bins=50, color='skyblue')
    plt.title('预测为正面评论的概率分布')
    plt.xlabel('预测为正面的概率')
    plt.ylabel('数量')
    plt.show()


























