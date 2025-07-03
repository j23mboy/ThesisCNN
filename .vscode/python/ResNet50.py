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
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import time

# 確認一下使用 cuda 或是 cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 使用 Focal Loss 並加入 class weights
# Define the FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

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
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def split_Train_Val_Data(data_dir):
    dataset = ImageFolder(data_dir) 
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
epochs = 15                                      # epoch 次數

data_dir = 'D:/資料集/root'                       # 資料夾名稱
fig_dir = 'D:/實驗結果/'                          # 圖片儲存的資料夾名稱

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)
C = models.resnet50(pretrained=True)
num_features = C.fc.in_features
C.fc = nn.Linear(num_features, 4)  # 將輸出調整為 4 個類別
C = C.to(device)
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer
summary(C, (3, 244, 244))                        # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
# 計算每個類別的權重
targets = [label for _, label in ImageFolder(data_dir).samples]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = FocalLoss(alpha=weights, gamma=2)  # gamma 可調整，默認為 2

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

# 訓練完成後儲存模型的路徑
model_save_path = 'D:/模型儲存/ResNer50_model.pth'

if __name__ == '__main__':
    code_start_time = time.time()    
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        # 初始化預測和真實標籤的列表
        all_predictions = []
        all_labels = []

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
            
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
            
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
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
                
                # 收集預測和真實標籤用於計算指標
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())       

        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * (correct_train / total_train)) # training accuracy
        test_acc.append(100 * (correct_test / total_test))    # testing accuracy
        loss_epoch_C.append((train_loss_C / iter))            # loss 

        # 計算並輸出 precision, recall, f1 score
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
    conf_matrix_pic_name = 'ResNet50_confusion_matrix_' + time.strftime("%Y%m%d", time.localtime()) + '.png'
    plt.savefig(os.path.join(fig_dir, conf_matrix_pic_name))
    plt.show()

code_end_time = time.time()
total_time = code_end_time - code_start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print('總花費時間： {:02}時{:02}分{:02}秒'.format(int(hours), int(minutes), int(seconds)))

# 將每一個 epoch 的 Loss 以及 Training / Testing accuracy 紀錄下來並繪製成圖
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

# 儲存訓練完成的模型
torch.save(C.state_dict(), model_save_path)
print(f"模型已儲存至 {model_save_path}")