import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle

class BaggingEnsembleClassifier:
    """
    集成分類器：使用 Bagging (Bootstrap Aggregating) 整合三個預訓練模型
    """
    
    def __init__(self, model_paths, test_data_path, n_bootstrap=3, sample_ratio=0.8):
        """
        初始化 Bagging 集成分類器
        
        Args:
            model_paths (dict): 包含三個模型路徑的字典
            test_data_path (str): 測試資料集路徑
            n_bootstrap (int): Bootstrap 採樣次數（通常等於模型數量）
            sample_ratio (float): 每次 Bootstrap 採樣的比例
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        self.n_bootstrap = n_bootstrap
        self.sample_ratio = sample_ratio
        self.models = {}
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用設備: {self.device}")
        print(f"Bagging 設定: Bootstrap 次數={n_bootstrap}, 採樣比例={sample_ratio}")
        
        # 先載入測試資料以獲得類別資訊
        self._get_class_info()

    def _get_class_info(self):
        """
        獲取類別資訊
        """
        try:
            from torchvision import datasets
            temp_dataset = datasets.ImageFolder(root=self.test_data_path)
            self.class_names = temp_dataset.classes
            print(f"偵測到 {len(self.class_names)} 個類別: {self.class_names}")
        except Exception as e:
            print(f"無法獲取類別資訊: {str(e)}")
            self.class_names = None
        
    def load_models(self):
        """
        載入三個預訓練模型
        """
        print("正在載入模型...")
        
        for model_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # 檢查載入的是否為狀態字典
                    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or all(isinstance(k, str) for k in checkpoint.keys())):
                        print(f"偵測到 {model_name} 為狀態字典格式，需要重建模型架構")
                        
                        # 根據模型名稱創建對應的模型架構
                        if model_name == 'ResNet34':
                            from torchvision.models import resnet34
                            model = resnet34(pretrained=False)
                            num_classes = len(self.class_names) if self.class_names else 10
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                        elif model_name == 'AlexNet':
                            from torchvision.models import alexnet
                            model = alexnet(pretrained=False)
                            num_classes = len(self.class_names) if self.class_names else 10
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                        elif model_name == 'VGG16':
                            from torchvision.models import vgg16
                            model = vgg16(pretrained=False)
                            num_classes = len(self.class_names) if self.class_names else 10
                            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                        else:
                            print(f"✗ 未知的模型類型: {model_name}")
                            continue
                        
                        # 載入狀態字典
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                    else:
                        model = checkpoint
                    
                    model.eval()
                    model = model.to(self.device)
                    self.models[model_name] = model
                    print(f"✓ 成功載入 {model_name}")
                    
                except Exception as e:
                    print(f"✗ 載入 {model_name} 失敗: {str(e)}")
            else:
                print(f"✗ 找不到模型檔案: {model_path}")
        
        if len(self.models) == 0:
            raise ValueError("沒有成功載入任何模型")
            
        print(f"總共載入了 {len(self.models)} 個模型")
    
    def load_test_data(self, batch_size=32):
        """
        載入測試資料集
        
        Args:
            batch_size (int): 批次大小
            
        Returns:
            test_loader: 測試資料載入器
        """
        print("正在載入測試資料...")
        
        # 定義資料轉換
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 載入測試資料集
        test_dataset = datasets.ImageFolder(
            root=self.test_data_path,
            transform=transform
        )
        
        # 創建資料載入器
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.class_names = test_dataset.classes
        print(f"發現 {len(self.class_names)} 個類別: {self.class_names}")
        print(f"測試樣本總數: {len(test_dataset)}")
        
        return test_loader
    
    def create_bootstrap_samples(self, test_loader):
        """
        創建 Bootstrap 採樣的數據子集
        
        Args:
            test_loader: 原始測試資料載入器
            
        Returns:
            list: Bootstrap 採樣的資料載入器列表
        """
        print(f"\n正在創建 {self.n_bootstrap} 個 Bootstrap 採樣...")
        
        dataset = test_loader.dataset
        total_samples = len(dataset)
        sample_size = int(total_samples * self.sample_ratio)
        
        bootstrap_loaders = []
        
        for i in range(self.n_bootstrap):
            # 有放回抽樣 (Bootstrap)
            indices = np.random.choice(total_samples, size=sample_size, replace=True)
            
            # 創建子集
            subset = Subset(dataset, indices)
            
            # 創建資料載入器
            loader = DataLoader(
                subset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            bootstrap_loaders.append(loader)
            print(f"Bootstrap 採樣 {i+1}: {len(indices)} 個樣本")
        
        return bootstrap_loaders
    
    def bagging_predict(self, test_loader):
        """
        使用 Bagging (多數投票) 進行集成預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            predictions: 集成預測結果
            true_labels: 真實標籤
        """
        print("\n正在進行 Bagging 預測 (多數投票)...")
        
        # 收集真實標籤
        true_labels = []
        for _, labels in test_loader:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        
        # 儲存每個模型的預測結果
        all_predictions = []
        
        # 對每個模型進行預測
        for model_name, model in self.models.items():
            print(f"正在使用 {model_name} 進行預測...")
            
            model_predictions = []
            model.eval()
            
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    model_predictions.extend(predicted.cpu().numpy())
            
            all_predictions.append(np.array(model_predictions))
        
        # Bagging: 使用多數投票決定最終預測
        all_predictions = np.array(all_predictions)  # shape: (n_models, n_samples)
        
        # 對每個樣本進行投票
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            # 找出得票最多的類別
            unique, counts = np.unique(votes, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            final_predictions.append(majority_vote)
        
        final_predictions = np.array(final_predictions)
        
        print("Bagging 預測完成 (使用多數投票)")
        
        return final_predictions, true_labels
    
    def export_misclassified_samples(self, test_loader, predicted_classes, true_labels, export_dir="D:/實驗結果/錯判樣本_Bagging"):
        """
        導出被錯誤分類的樣本圖片
        """
        print("\n正在導出錯誤分類的樣本圖片...")
        
        os.makedirs(export_dir, exist_ok=True)
        
        dataset = test_loader.dataset
        samples = dataset.samples
        
        misclassified_indices = np.where(predicted_classes != true_labels)[0]
        total_misclassified = len(misclassified_indices)
        
        if total_misclassified == 0:
            print("沒有錯誤分類的樣本！")
            return 0
        
        print(f"發現 {total_misclassified} 個錯誤分類的樣本")
        
        from PIL import Image
        import shutil
        
        for idx in misclassified_indices:
            img_path, _ = samples[idx]
            true_label = true_labels[idx]
            pred_label = predicted_classes[idx]
            
            true_class_name = self.class_names[true_label]
            pred_class_name = self.class_names[pred_label]
            
            error_class_dir = os.path.join(export_dir, f"{true_class_name}_誤判為_{pred_class_name}")
            os.makedirs(error_class_dir, exist_ok=True)
            
            filename = os.path.basename(img_path)
            dest_path = os.path.join(error_class_dir, filename)
            
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"無法複製圖片 {img_path}: {str(e)}")
        
        print(f"錯誤分類的樣本已導出至: {export_dir}")
        
        return total_misclassified
    
    def evaluate_performance(self, predicted_classes, true_labels):
        """
        評估模型性能
        """
        print("\n" + "="*50)
        print("Bagging 模型評估結果")
        print("="*50)
        
        accuracy = accuracy_score(true_labels, predicted_classes)
        print(f"整體準確率 (Accuracy): {accuracy:.4f}")
        
        print("\n分類報告 (Classification Report):")
        print("-" * 50)
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        return accuracy, report
    
    def plot_confusion_matrix(self, true_labels, predicted_classes, save_path=None):
        """
        繪製混淆矩陣
        """
        print("\n正在生成混淆矩陣...")
        
        cm = confusion_matrix(true_labels, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - Ensemble Model (Bagging)')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩陣已儲存至: {save_path}")
        
        plt.show()
        
        return cm
    
    def save_ensemble_model(self, save_dir="D:/模型儲存/ensemble_models"):
        """
        儲存 Bagging 集成模型配置
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if not os.access(save_dir, os.W_OK):
                save_dir = os.path.join(os.getcwd(), "ensemble_models")
                os.makedirs(save_dir, exist_ok=True)
                print(f"改用當前目錄: {save_dir}")
            
            bagging_config = {
                'model_type': 'bagging',
                'n_bootstrap': self.n_bootstrap,
                'sample_ratio': self.sample_ratio,
                'class_names': self.class_names,
                'model_paths': self.model_paths,
                'device': str(self.device),
                'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())
            }
            
            config_path = os.path.join(save_dir, "bagging_config.pkl")
            with open(config_path, 'wb') as f:
                pickle.dump(bagging_config, f)
            
            print(f"Bagging 集成模型配置已儲存至: {config_path}")
            return save_dir
            
        except Exception as e:
            print(f"儲存 Bagging 集成模型失敗: {str(e)}")
            return None
    
    def load_ensemble_model(self, load_dir="D:/模型儲存/ensemble_models"):
        """
        載入 Bagging 集成模型配置
        """
        try:
            config_path = os.path.join(load_dir, "bagging_config.pkl")
            
            if not os.path.exists(config_path):
                backup_dir = os.path.join(os.getcwd(), "ensemble_models")
                config_path = os.path.join(backup_dir, "bagging_config.pkl")
                
                if not os.path.exists(config_path):
                    print(f"找不到 Bagging 配置檔案: {config_path}")
                    return False
            
            with open(config_path, 'rb') as f:
                bagging_config = pickle.load(f)
            
            self.n_bootstrap = bagging_config['n_bootstrap']
            self.sample_ratio = bagging_config['sample_ratio']
            self.class_names = bagging_config['class_names']
            
            print(f"Bagging 集成模型配置已載入: {config_path}")
            print(f"Bootstrap 次數: {self.n_bootstrap}")
            print(f"採樣比例: {self.sample_ratio}")
            
            return True
            
        except Exception as e:
            print(f"載入 Bagging 集成模型失敗: {str(e)}")
            return False

    def run_ensemble_evaluation(self):
        """
        執行完整的 Bagging 集成評估流程
        """
        try:
            # 1. 載入模型
            self.load_models()
            
            # 2. 載入測試資料
            test_loader = self.load_test_data()
            
            # 3. 進行 Bagging 預測
            predicted_classes, true_labels = self.bagging_predict(test_loader)
            
            # 4. 評估性能
            accuracy, report = self.evaluate_performance(predicted_classes, true_labels)
            
            # 5. 繪製混淆矩陣
            confusion_matrix_path = "D:/實驗結果/confusion_matrix_bagging_" + time.strftime("%Y%m%d", time.localtime()) + '.png'
            cm = self.plot_confusion_matrix(true_labels, predicted_classes, confusion_matrix_path)
            
            # 6. 導出錯誤分類的樣本
            export_dir = "D:/實驗結果/錯判樣本_Bagging_" + time.strftime("%Y%m%d", time.localtime())
            misclassified_count = self.export_misclassified_samples(test_loader, predicted_classes, true_labels, export_dir)
            
            # 7. 儲存 Bagging 集成模型（新增）
            try:
                save_dir = f"D:/模型儲存/ensemble_bagging_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
                saved_path = self.save_ensemble_model(save_dir)
                if saved_path:
                    print(f"Bagging 集成模型已儲存至: {saved_path}")
            except Exception as save_error:
                print(f"儲存 Bagging 集成模型時發生錯誤: {str(save_error)}")
            
            print("\n" + "="*50)
            print("Bagging 集成評估完成！")
            print("="*50)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': predicted_classes,
                'true_labels': true_labels,
                'misclassified_count': misclassified_count,
                'saved_model_path': saved_path if 'saved_path' in locals() else None
            }
            
        except Exception as e:
            print(f"執行過程中發生錯誤: {str(e)}")
            return None

def main():
    """
    主函數
    """
    print("Bagging Ensemble 模型評估系統 (PyTorch)")
    print("="*50)
    
    # 定義模型路徑
    model_paths = {
        'ResNet34': 'D:/模型儲存/ResNet34_model.pth',
        'AlexNet': 'D:/模型儲存/AlexNet_model.pth',
        'VGG16': 'D:/模型儲存/VGG16_model.pth'
    }
    
    # 定義測試資料路徑
    test_data_path = 'D:/資料集/root'
    
    # 創建 Bagging 集成分類器
    bagging_classifier = BaggingEnsembleClassifier(
        model_paths=model_paths, 
        test_data_path=test_data_path,
        n_bootstrap=3,  # Bootstrap 採樣次數
        sample_ratio=0.8  # 每次採樣 80% 的數據
    )
    
    # 執行評估
    results = bagging_classifier.run_ensemble_evaluation()
    
    if results:
        print(f"\n最終準確率: {results['accuracy']:.4f}")
        print(f"錯誤分類樣本數: {results['misclassified_count']}")
    else:
        print("評估失敗，請檢查模型和資料路徑是否正確")

if __name__ == "__main__":
    main()
