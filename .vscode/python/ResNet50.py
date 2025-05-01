# import libraries
import os, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
from sklearn.metrics import recall_score, precision_score, f1_score

import time

# 確認一下使用 cuda 或是 cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 實作一個可以讀取的Pytorch dataset
class DogDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.transform = transform    # 影像的轉換方式
 
    def __len__(self):
        return len(self.filenames)    # return DataSet 長度
 
    def __getitem__(self, idx):       # idx: Inedx of filenames
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image) # Transform image
        label = np.array(self.labels[idx])
        return image, label           # return 模型訓練所需的資訊

# 定義 Normalize 以及 Transform 的參數
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def split_Train_Val_Data(data_dir):
    dataset = ImageFolder(data_dir) 
    # 建立 20 類的 list
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        np.random.seed(42)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------
        num_sample_train = int(len(data) * 0.8)
        num_sample_test = len(data) - num_sample_train
        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))
        
        for x in data[:num_sample_train] : # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[num_sample_train:] : # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(DogDataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(DogDataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader

# 參數設定
batch_size = 64                                  # Batch Size
lr = 1e-3                                        # Learning Rate
epochs = 10                                      # epoch 次數

data_dir = 'D:/資料集/root'                       # 資料夾名稱

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)
C = models.resnet50(pretrained=True).to(device)     # 使用內建的 model 
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer
summary(C, (3, 244, 244))                        # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
# Loss function
criterion = nn.CrossEntropyLoss()                # 選擇想用的 loss function

loss_epoch_C = []
train_acc, test_acc = [], []
recall_epoch, precision_epoch, f1_epoch = [], [], []
best_acc, best_auc = 0.0, 0.0

if __name__ == '__main__':
    code_start_time = time.time()    
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        # 初始化 True Positive 和 True Negative 的計數器
        true_positive, true_negative = 0, 0
        false_positive, false_negative = 0, 0

        C.train() # 設定 train 或 eval
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
        
        # ---------------------------
        # Training Stage
        # ---------------------------
        for i, (x, label) in enumerate(train_dataloader) :
            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()                         # 清空梯度
            train_output = C(x)                             # 將訓練資料輸入至模型進行訓練 (Forward propagation)
            train_loss = criterion(train_output, label)     # 計算 loss
            train_loss.backward()                           # 將 loss 反向傳播
            optimizer_C.step()                              # 更新權重
            
            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(train_output.data, 1)  # 取出預測的 maximum
            # 獲取所有可能的類別
            class_names = torch.unique(torch.cat((predicted, label))).tolist()
            # 初始化 TP、TN、FP、FN
            TP = {c: 0 for c in class_names}
            TN = {c: 0 for c in class_names}
            FP = {c: 0 for c in class_names}
            FN = {c: 0 for c in class_names}

            for c in class_names:
                TP[c] = ((predicted == c) & (label == c)).sum().item()
                TN[c] = ((predicted != c) & (label != c)).sum().item()
                FP[c] = ((predicted == c) & (label != c)).sum().item()
                FN[c] = ((predicted != c) & (label == c)).sum().item()

                correct_train += (TP[c] + TN[c])
                total_train += (TP[c] + TN[c] + FP[c] + FN[c])
            # total_train += label.size(0)
            # correct_train += (predicted == label).sum()
            
            train_loss_C += train_loss.item()
            iter += 1
                    
        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iter, correct_train / total_train))
        
        # --------------------------
        # Testing Stage
        # --------------------------
        C.eval() # 設定 train 或 eval
        for i, (x, label) in enumerate(test_dataloader) :
            with torch.no_grad():                           # 測試階段不需要求梯度
                x, label = x.to(device), label.to(device)
                test_output = C(x)                          # 將測試資料輸入至模型進行測試
                test_loss = criterion(test_output, label)   # 計算 loss
                
                # 計算測試資料的準確度 (correct_test / total_test)
                _, predicted = torch.max(test_output.data, 1)
                # total_test += label.size(0)
                # correct_test += (predicted == label).sum()
                
                # 獲取所有可能的類別
                class_names = torch.unique(torch.cat((predicted, label))).tolist()

                # 初始化 TP、TN、FP、FN
                TP = {c: 0 for c in class_names}
                TN = {c: 0 for c in class_names}
                FP = {c: 0 for c in class_names}
                FN = {c: 0 for c in class_names}

                for c in class_names:
                    TP[c] = ((predicted == c) & (label == c)).sum().item()
                    TN[c] = ((predicted != c) & (label != c)).sum().item()
                    FP[c] = ((predicted == c) & (label != c)).sum().item()
                    FN[c] = ((predicted != c) & (label == c)).sum().item()

                    true_positive += TP[c]
                    true_negative += TN[c]
                    false_positive += FP[c]
                    false_negative += FN[c]

                    correct_test += (TP[c] + TN[c])
                    total_test += (TP[c] + TN[c] + FP[c] + FN[c])
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * (correct_train / total_train)) # training accuracy
        test_acc.append(100 * (correct_test / total_test))    # testing accuracy
        loss_epoch_C.append((train_loss_C / iter))            # loss 
        
        print('true_positive: %d | true_negative: %d | false_positive: %d | false_negative: %d' % \
              (true_positive, true_negative, false_positive, false_negative))
        if true_positive == 0:
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = true_positive / (true_positive + false_negative)
            precision = true_positive / (true_positive + false_positive)
            f1 = 2 * (precision * recall) / (precision + recall)
        print('Recall: %.3f | Precision: %.3f | F1: %.3f' % (recall, precision, f1))
        recall_epoch.append(recall)
        precision_epoch.append(precision)
        f1_epoch.append(f1)

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))

code_end_time = time.time()
print('總花費時間： %.3f(secs)' % (code_end_time - code_start_time))

# 將每一個 epoch 的 Loss 以及 Training / Testing accuracy 紀錄下來並繪製成圖
fig_dir = 'D:/實驗結果/'
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# 取得現在時間，格式為yyyyMMdd
now = time.strftime("%Y%m%d", time.localtime())

loss_pic_name = 'ResNet50_loss_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, loss_pic_name))
plt.show()

acc_pic_name = 'ResNet50_acc_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, acc_pic_name))
plt.show()

recall_pic_name = 'ResNet50_recall_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), recall_epoch)    # plot your recall
plt.title('Training recall')
plt.ylabel('recall'), plt.xlabel('epoch')
plt.legend(['recall'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, recall_pic_name))
plt.show()

precision_pic_name = 'ResNet50_precision_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), precision_epoch)    # plot your precision
plt.title('Training precision')
plt.ylabel('precision'), plt.xlabel('epoch')
plt.legend(['precision'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, precision_pic_name))
plt.show()

f1_pic_name = 'ResNet50_f1_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), f1_epoch)    # plot your f1
plt.title('Training f1')
plt.ylabel('f1'), plt.xlabel('epoch')
plt.legend(['f1'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, f1_pic_name))
plt.show()
