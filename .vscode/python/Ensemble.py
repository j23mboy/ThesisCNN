import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import time
import pickle

class EnsembleClassifier:
    """
    集成分類器：使用 Soft Voting 整合三個預訓練模型
    """
    
    def __init__(self, model_paths, test_data_path, model_weights=None):
        """
        初始化集成分類器
        
        Args:
            model_paths (dict): 包含三個模型路徑的字典
            test_data_path (str): 測試資料集路徑
            model_weights (dict): 每個模型的權重，如果為None則使用平均權重
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        self.models = {}
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_weights = {'ResNet34': 0.4,'AlexNet': 0.3,'VGG16': 0.3}

        # 設定模型權重
        if model_weights is None:
            # 預設使用平均權重
            self.model_weights = {name: 1.0 for name in model_paths.keys()}
        else:
            self.model_weights = model_weights
        
        # 正規化權重（確保總和為1）
        total_weight = sum(self.model_weights.values())
        self.model_weights = {name: weight/total_weight for name, weight in self.model_weights.items()}
        
        print(f"使用設備: {self.device}")
        print(f"模型權重: {self.model_weights}")
        
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
                    # 先嘗試載入完整模型
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # 檢查載入的是否為狀態字典
                    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or all(isinstance(k, str) for k in checkpoint.keys())):
                        print(f"偵測到 {model_name} 為狀態字典格式，需要重建模型架構")
                        
                        # 根據模型名稱創建對應的模型架構
                        if model_name == 'ResNet34':
                            from torchvision.models import resnet34
                            model = resnet34(pretrained=False)
                            # 根據你的類別數修改最後一層
                            num_classes = len(self.class_names) if self.class_names else 10  # 預設10類
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
                        # 如果是完整模型
                        model = checkpoint
                    
                    model.eval()  # 設置為評估模式
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
            img_size (tuple): 圖片尺寸
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
                               std=[0.229, 0.224, 0.225])  # ImageNet 標準化
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
            shuffle=False,  # 不打亂順序，確保預測結果對應正確
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 儲存類別名稱
        self.class_names = test_dataset.classes
        print(f"發現 {len(self.class_names)} 個類別: {self.class_names}")
        print(f"測試樣本總數: {len(test_dataset)}")
        
        return test_loader
    
    def soft_voting_predict(self, test_loader):
        """
        使用加權 Soft Voting 進行集成預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            predictions: 集成預測結果
            true_labels: 真實標籤
        """
        print("正在進行加權 Soft Voting 預測...")
        print(f"使用權重: {self.model_weights}")
        
        all_predictions = []
        true_labels = []
        
        # 收集真實標籤
        for _, labels in test_loader:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        
        # 收集所有模型的預測結果
        for model_name, model in self.models.items():
            print(f"正在使用 {model_name} 進行預測（權重: {self.model_weights[model_name]:.3f}）...")
            
            model_predictions = []
            model.eval()
            
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    model_predictions.extend(probabilities.cpu().numpy())
            
            # 將預測結果乘以對應的權重
            weighted_predictions = np.array(model_predictions) * self.model_weights[model_name]
            all_predictions.append(weighted_predictions)
        
        # 計算加權 Soft Voting（加權平均所有模型的預測機率）
        ensemble_predictions = np.sum(all_predictions, axis=0)
        
        # 獲取預測類別
        predicted_classes = np.argmax(ensemble_predictions, axis=1)
        
        print("加權 Soft Voting 預測完成")
        
        return predicted_classes, true_labels, ensemble_predictions
    
    def export_misclassified_samples(self, test_loader, predicted_classes, true_labels, export_dir="D:/實驗結果/錯判樣本"):
        """
        導出被錯誤分類的樣本圖片
        
        Args:
            test_loader: 測試資料載入器
            predicted_classes: 預測類別
            true_labels: 真實標籤
            export_dir: 導出目錄
        """
        print("\n正在導出錯誤分類的樣本圖片...")
        
        # 確保導出目錄存在
        os.makedirs(export_dir, exist_ok=True)
        
        # 取得測試資料集的圖片路徑
        dataset = test_loader.dataset
        samples = dataset.samples  # 包含 (path, class_idx) 的列表
        
        # 尋找錯誤分類的樣本
        misclassified_indices = np.where(predicted_classes != true_labels)[0]
        total_misclassified = len(misclassified_indices)
        
        if total_misclassified == 0:
            print("沒有錯誤分類的樣本！")
            return
        
        print(f"發現 {total_misclassified} 個錯誤分類的樣本")
        
        # 對於每個錯誤分類的樣本
        from PIL import Image
        import shutil
        
        for idx in misclassified_indices:
            # 獲取原始圖片路徑和真實標籤
            img_path, _ = samples[idx]
            true_label = true_labels[idx]
            pred_label = predicted_classes[idx]
            
            # 獲取類別名稱
            true_class_name = self.class_names[true_label]
            pred_class_name = self.class_names[pred_label]
            
            # 創建錯誤分類目錄 (如果不存在)
            error_class_dir = os.path.join(export_dir, f"{true_class_name}_誤判為_{pred_class_name}")
            os.makedirs(error_class_dir, exist_ok=True)
            
            # 複製圖片到錯誤分類目錄
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
        
        Args:
            predicted_classes: 預測類別
            true_labels: 真實標籤
        """
        print("\n" + "="*50)
        print("模型評估結果")
        print("="*50)
        
        # 計算準確率
        accuracy = accuracy_score(true_labels, predicted_classes)
        print(f"整體準確率 (Accuracy): {accuracy:.4f}")
        
        # 產生分類報告
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
        
        Args:
            true_labels: 真實標籤
            predicted_classes: 預測類別
            save_path: 儲存路徑（可選）
        """
        print("\n正在生成混淆矩陣...")
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # 繪製混淆矩陣
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - Ensemble Model (Soft Voting)')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩陣已儲存至: {save_path}")
        
        plt.show()
        
        return cm
    
    def validate_image_files(self):
        """
        驗證圖片檔案的完整性
        """
        print("正在驗證圖片檔案...")
        
        from PIL import Image
        import os
        
        corrupted_files = []
        
        for root, dirs, files in os.walk(self.test_data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    try:
                        # 嘗試開啟圖片
                        with Image.open(file_path) as img:
                            img.verify()  # 驗證圖片
                    except Exception as e:
                        print(f"損壞的圖片檔案: {file_path} - {str(e)}")
                        corrupted_files.append(file_path)
        
        if corrupted_files:
            print(f"發現 {len(corrupted_files)} 個損壞的圖片檔案")
            # 可選：移除或移動損壞的檔案
            return False
        else:
            print("所有圖片檔案驗證通過")
            return True

    def save_ensemble_model(self, save_dir="D:/模型儲存/ensemble_models"):
        """
        儲存完整的集成模型配置
        
        Args:
            save_dir (str): 儲存目錄
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 檢查目錄是否可寫入
            if not os.access(save_dir, os.W_OK):
                save_dir = os.path.join(os.getcwd(), "ensemble_models")
                os.makedirs(save_dir, exist_ok=True)
                print(f"改用當前目錄: {save_dir}")
            
            # 儲存集成模型配置
            ensemble_config = {
                'model_type': 'weighted_soft_voting',
                'model_weights': self.model_weights,
                'class_names': self.class_names,
                'model_paths': self.model_paths,
                'device': str(self.device),
                'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())
            }
            
            config_path = os.path.join(save_dir, "ensemble_weights.pkl")
            with open(config_path, 'wb') as f:
                pickle.dump(ensemble_config, f)
            
            print(f"集成模型配置已儲存至: {config_path}")
            
            # 儲存模型路徑資訊（用於重新載入基礎模型）
            paths_info = {
                'base_models': self.model_paths,
                'ensemble_type': 'soft_voting',
                'weights': self.model_weights
            }
            
            paths_path = os.path.join(save_dir, "model_paths_info.pkl")
            with open(paths_path, 'wb') as f:
                pickle.dump(paths_info, f)
            
            print(f"模型路徑資訊已儲存至: {paths_path}")
            
            return save_dir
            
        except Exception as e:
            print(f"儲存集成模型失敗: {str(e)}")
            return None
    
    def load_ensemble_model(self, load_dir="D:/模型儲存/ensemble_models"):
        """
        載入完整的集成模型配置
        
        Args:
            load_dir (str): 載入目錄
            
        Returns:
            bool: 載入是否成功
        """
        try:
            config_path = os.path.join(load_dir, "ensemble_weights.pkl")
            
            if not os.path.exists(config_path):
                # 嘗試在當前目錄尋找
                backup_dir = os.path.join(os.getcwd(), "ensemble_models")
                config_path = os.path.join(backup_dir, "ensemble_weights.pkl")
                
                if not os.path.exists(config_path):
                    print(f"找不到集成模型配置檔案: {config_path}")
                    return False
            
            # 載入集成模型配置
            with open(config_path, 'rb') as f:
                ensemble_config = pickle.load(f)
            
            self.model_weights = ensemble_config['model_weights']
            self.class_names = ensemble_config['class_names']
            
            print(f"集成模型配置已載入: {config_path}")
            print(f"載入的權重: {self.model_weights}")
            print(f"類別名稱: {self.class_names}")
            
            return True
            
        except Exception as e:
            print(f"載入集成模型失敗: {str(e)}")
            return False
    
    def predict_with_saved_model(self, test_loader):
        """
        使用已儲存的集成模型進行預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            tuple: 預測結果
        """
        if not self.models:
            print("基礎模型尚未載入，正在載入...")
            self.load_models()
        
        if not self.model_weights:
            print("集成權重尚未設定，請先載入集成模型配置")
            return None
        
        # 使用現有的預測方法
        return self.soft_voting_predict(test_loader)

    def run_ensemble_evaluation(self):
        """
        執行完整的集成評估流程
        """
        try:
            # 0. 驗證圖片檔案（新增）
            if not self.validate_image_files():
                print("警告：發現損壞的圖片檔案，建議先處理")
            
            # 1. 載入模型
            self.load_models()
            
            # 2. 載入測試資料
            test_loader = self.load_test_data()
            
            # 3. 進行 Soft Voting 預測
            predicted_classes, true_labels, ensemble_predictions = self.soft_voting_predict(test_loader)
            
            # 4. 評估性能
            accuracy, report = self.evaluate_performance(predicted_classes, true_labels)
            
            # 5. 繪製混淆矩陣
            confusion_matrix_path = "D:/實驗結果/confusion_matrix_ensemble" + time.strftime("%Y%m%d", time.localtime()) + '.png'
            cm = self.plot_confusion_matrix(true_labels, predicted_classes, confusion_matrix_path)
            
            # 6. 導出錯誤分類的樣本
            export_dir = "D:/實驗結果/錯判樣本_" + time.strftime("%Y%m%d", time.localtime())
            misclassified_count = self.export_misclassified_samples(test_loader, predicted_classes, true_labels, export_dir)
            
            # 7. 儲存集成模型配置（新增）
            try:
                save_dir = f"D:/模型儲存/ensemble_soft_voting_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
                saved_path = self.save_ensemble_model(save_dir)
                if saved_path:
                    print(f"集成模型已儲存至: {saved_path}")
            except Exception as save_error:
                print(f"儲存集成模型時發生錯誤: {str(save_error)}")
            
            print("\n" + "="*50)
            print("集成評估完成！")
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
            import traceback
            traceback.print_exc()
            return None

def main():
    """
    主函數
    """
    print("Ensemble 模型評估系統 (PyTorch)")
    print("="*50)
    
    # 定義模型路徑 (現在應該是 .pth 或 .pt 檔案)
    model_paths = {
        'ResNet34': 'D:/模型儲存/ResNet34_model.pth',
        'AlexNet': 'D:/模型儲存/AlexNet_model.pth',
        'VGG16': 'D:/模型儲存/VGG16_model.pth'
    }
    
    # 定義測試資料路徑
    test_data_path = 'D:/資料集/root'
    
    # 創建集成分類器
    ensemble_classifier = EnsembleClassifier(model_paths, test_data_path)
    
    # 執行評估
    results = ensemble_classifier.run_ensemble_evaluation()
    
    if results:
        print(f"\n最終準確率: {results['accuracy']:.4f}")
    else:
        print("評估失敗，請檢查模型和資料路徑是否正確")

if __name__ == "__main__":
    main()
