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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

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
    character = [[] for i in range(len(dataset.classes))]
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        np.random.seed(42)
        np.random.shuffle(data)
            
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        num_sample_train = int(len(data) * 0.8)
        num_sample_test = len(data) - num_sample_train
        
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
batch_size = 32                 # Batch Size
lr = 1e-3                       # Learning Rate
epochs = 10                     # epoch 次數
data_dir = 'D:/資料集/root'      # 資料夾名稱
fig_dir = 'D:/實驗結果/'         # 圖片儲存的資料夾名稱

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)

# 使用預訓練的 EfficientNet-B0 模型
model = models.efficientnet_b0(pretrained=True)
# 修改最後一層分類層以適應我們的類別數
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)  # 將輸出調整為 4 個類別
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam優化器
# 添加學習率調整器，當測試損失連續4個epoch未改善時，學習率減半
scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5, cooldown=1, threshold=0.003, mode='max', verbose=True)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵損失函數

summary(model, (3, 224, 224))  # 輸出模型結構摘要

loss_epoch = []
train_acc, test_acc = [], []

if __name__ == '__main__': 
    code_start_time = time.time()
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss = 0.0

        # 初始化預測和真實標籤的列表
        all_predictions = []
        all_labels = []

        model.train()  # 設定為訓練模式
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
        
        # 訓練階段
        for i, (x, label) in enumerate(train_dataloader):
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            train_output = model(x)
            loss = criterion(train_output, label)
            loss.backward()
            optimizer.step()
            
            # 計算訓練資料的準確度
            _, predicted = torch.max(train_output.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
            
            train_loss += loss.item()
            iter += 1
                    
        print('Training epoch: %d / loss: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss / iter, correct_train / total_train))
        
        # 測試階段
        model.eval()  # 設定為評估模式
        test_loss = 0.0
        
        with torch.no_grad():
            for i, (x, label) in enumerate(test_dataloader):
                x, label = x.to(device), label.to(device)
                test_output = model(x)
                test_loss += criterion(test_output, label).item()
                
                _, predicted = torch.max(test_output.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
                
                # 收集預測和真實標籤用於計算指標
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
         # 使用平均測試損失來調整學習率
        avg_test_loss = test_loss / len(test_dataloader)
        scheduler.step(avg_test_loss)
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * (correct_train / total_train))  # training accuracy
        test_acc.append(100 * (correct_test / total_test))     # testing accuracy
        loss_epoch.append((train_loss / iter))                 # loss 

        # 計算並輸出 precision, recall, f1 score
        # 獲取所有可能的類別
        class_names = np.unique(np.concatenate((all_predictions, all_labels)))
        precision = precision_score(all_labels, all_predictions, average=None)
        recall = recall_score(all_labels, all_predictions, average=None)
        f1 = f1_score(all_labels, all_predictions, average=None)
        for i, class_name in enumerate(class_names):
            print(f"Class {class_name}: Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1 Score: {f1[i]:.3f}")

        print("Unique labels:", np.unique(all_labels))
        print("Unique predictions:", np.unique(all_predictions))

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))
    
    # 繪製最後一個 epoch 的混淆矩陣
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    # 保存混淆矩陣圖
    conf_matrix_pic_name = 'EfficientNet_confusion_matrix_' + time.strftime("%Y%m%d", time.localtime()) + '.png'
    plt.savefig(os.path.join(fig_dir, conf_matrix_pic_name))
    plt.show()

code_end_time = time.time()
total_time = code_end_time - code_start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print('總花費時間： {:02}時{:02}分{:02}秒'.format(int(hours), int(minutes), int(seconds)))

# 取得現在時間，格式為yyyyMMdd
now = time.strftime("%Y%m%d", time.localtime())

loss_pic_name = 'EfficientNet_loss_' + now + '.png'
# 繪製loss曲線
plt.figure()
plt.plot(list(range(epochs)), loss_epoch)
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, loss_pic_name))
plt.show()

# 繪製準確率曲線
acc_pic_name = 'EfficientNet_acc_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), train_acc)
plt.plot(list(range(epochs)), test_acc)
plt.title('Accuracy')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc='upper left')
plt.savefig(os.path.join(fig_dir, acc_pic_name))
plt.show()

# 儲存訓練完成的模型
# model_save_path = 'D:/模型儲存/EfficientNet_model.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f"模型已儲存至 {model_save_path}")
