# import libraries
import os, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    transforms.RandomResizedCrop(256),
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
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        # np.random.seed(42)
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
batch_size = 32                                  # Batch Size
lr = 1e-3                                        # Learning Rate
epochs = 20                                      # epoch 次數

data_dir = 'D:/資料集/root'                       # 資料夾名稱
fig_dir = 'D:/實驗結果/'                          # 圖片儲存的資料夾名稱

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)
C = models.resnet34(pretrained=True)
num_features = C.fc.in_features
C.fc = nn.Linear(num_features, 4)  # 將輸出調整為 4 個類別
C = C.to(device)
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer
summary(C, (3, 244, 244))                        # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
# 計算每個類別的權重
class_counts = np.bincount([label for _, label in ImageFolder(data_dir).samples])
class_weights = 1.0 / class_counts
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss function with class weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.25, reduction='mean'):  # 更新 gamma 值為 1.5
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用 Focal Loss 替代 Label Smoothing
criterion = FocalLoss(alpha=1, gamma=1.25, reduction='mean')

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

scheduler = ReduceLROnPlateau(optimizer_C, mode='min', factor=0.1, patience=3, threshold=0.001, verbose=True)

# Early Stopping 機制
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=5, verbose=True)

# 訓練完成後儲存模型的路徑
model_save_path = 'D:/模型儲存/ResNet34_model.pth'

if __name__ == '__main__':
    code_start_time = time.time()
    # 儲存一個真正的副本，而不是參考
    import copy
    original_test_dataloader = copy.deepcopy(test_dataloader)
    
    # 獲取所有類別名稱
    class_names = ImageFolder(data_dir).classes
    
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        # 初始化預測和真實標籤的列表
        all_predictions = []
        all_labels = []
        # 用於保存錯誤樣本的列表
        misclassified_samples = []

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
                
                # 收集錯誤分類的樣本
                if epoch == epochs - 1 or early_stopping.counter >= early_stopping.patience - 1:  # 只在最後一個epoch收集
                    misclassified_indices = (predicted != label).nonzero(as_tuple=True)[0]
                    for idx in misclassified_indices:
                        # 收集原始文件路徑、真實標籤和預測標籤
                        orig_idx = i * batch_size + idx.item()
                        if orig_idx < len(test_dataloader.dataset.filenames):
                            file_path = test_dataloader.dataset.filenames[orig_idx]
                            true_label = label[idx].item()
                            pred_label = predicted[idx].item()
                            misclassified_samples.append((file_path, true_label, pred_label))

        print('Testing acc: %.3f' % (correct_test / total_test))
        
        # 更新學習率
        scheduler.step(train_loss_C / iter)
        
        # Early Stopping 判斷
        early_stopping(train_loss_C / iter)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
                                     
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

    # 函數用於導出錯誤分類的樣本
    def export_misclassified_samples(misclassified_samples, class_names):
        now = time.strftime("%Y%m%d", time.localtime())
        misclassified_dir = os.path.join(fig_dir, f'錯判樣本_ResNet34_{now}')
        
        if not os.path.exists(misclassified_dir):
            os.makedirs(misclassified_dir)
            
        print(f"正在導出 {len(misclassified_samples)} 個錯誤分類樣本到 {misclassified_dir}...")
        
        # 為每個類別創建子目錄
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(misclassified_dir, f"{class_idx}_{class_name}")
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
        
        for idx, (file_path, true_label, pred_label) in enumerate(misclassified_samples):
            # 獲取原始檔案名
            orig_filename = os.path.basename(file_path)
            
            # 創建新的檔案名，包含真實標籤和預測標籤
            new_filename = f"true_{class_names[true_label]}_pred_{class_names[pred_label]}_{orig_filename}"
            
            # 存放到真實標籤的目錄下
            target_dir = os.path.join(misclassified_dir, f"{true_label}_{class_names[true_label]}")
            target_path = os.path.join(target_dir, new_filename)
            
            # 複製檔案
            shutil.copy2(file_path, target_path)
        
        print(f"錯誤分類樣本導出完成！共 {len(misclassified_samples)} 個樣本")
    
    # 導出最後一個 epoch 的錯誤分類樣本
    if misclassified_samples:
        export_misclassified_samples(misclassified_samples, class_names)
    
    # 繪製最後一個 epoch 的混淆矩陣
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    # 保存混淆矩陣圖
    conf_matrix_pic_name = 'ResNet34_confusion_matrix_' + time.strftime("%Y%m%d", time.localtime()) + '.png'
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

loss_pic_name = 'ResNet34_loss_' + now + '.png'
plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig(os.path.join(fig_dir, loss_pic_name))
plt.show()

acc_pic_name = 'ResNet34_acc_' + now + '.png'
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