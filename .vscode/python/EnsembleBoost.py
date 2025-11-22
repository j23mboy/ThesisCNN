import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle

class BoostingEnsembleClassifier:
    """
    集成分類器：使用自定義 Boosting 整合三個預訓練模型
    基於錯誤率動態調整模型權重，使用加權 Soft Voting
    """
    
    def __init__(self, model_paths, test_data_path, weight_train_ratio=0.3, n_iterations=5, learning_rate=0.1):
        """
        初始化 Boosting 集成分類器
        
        Args:
            model_paths (dict): 包含三個模型路徑的字典
            test_data_path (str): 測試資料集路徑
            weight_train_ratio (float): 用於權重學習的測試資料比例
            n_iterations (int): Boosting 迭代次數
            learning_rate (float): 權重更新學習率
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        self.weight_train_ratio = weight_train_ratio
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.models = {}
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型權重（平均分配）
        self.model_weights = {name: 1.0/len(model_paths) for name in model_paths.keys()}
        self.weight_history = []  # 記錄權重變化歷史
        self.error_history = []   # 記錄錯誤率歷史
        
        print(f"使用設備: {self.device}")
        print(f"Boosting 設定:")
        print(f"  - 權重學習資料比例: {weight_train_ratio:.0%}")
        print(f"  - Boosting 迭代次數: {n_iterations}")
        print(f"  - 學習率: {learning_rate}")
        print(f"  - 初始模型權重: {self.model_weights}")
        
        # 先載入測試資料以獲得類別資訊
        self._get_class_info()

    def _get_class_info(self):
        """
        獲取類別資訊
        """
        try:
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
    
    def load_data(self, batch_size=32):
        """
        載入資料集，分割權重學習和最終評估資料
        
        Returns:
            tuple: (weight_train_loader, final_test_loader)
        """
        print("正在載入資料...")
        
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
        
        # 分層採樣分割資料：30% 用於權重學習，70% 用於最終評估
        labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        indices = list(range(len(test_dataset)))
        
        weight_train_indices, final_test_indices = train_test_split(
            indices,
            test_size=1-self.weight_train_ratio,
            stratify=labels,
            random_state=42
        )
        
        # 創建子集
        weight_train_subset = Subset(test_dataset, weight_train_indices)
        final_test_subset = Subset(test_dataset, final_test_indices)
        
        # 創建資料載入器
        weight_train_loader = DataLoader(
            weight_train_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        final_test_loader = DataLoader(
            final_test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.class_names = test_dataset.classes
        print(f"權重學習樣本數: {len(weight_train_subset)}")
        print(f"最終測試樣本數: {len(final_test_subset)}")
        print(f"發現 {len(self.class_names)} 個類別: {self.class_names}")
        
        return weight_train_loader, final_test_loader
    
    def get_model_predictions(self, data_loader):
        """
        獲取所有模型對資料的預測結果
        
        Args:
            data_loader: 資料載入器
            
        Returns:
            tuple: (model_predictions, true_labels)
        """
        # 收集真實標籤
        true_labels = []
        for _, labels in data_loader:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        
        # 收集每個模型的預測結果
        model_predictions = {}
        model_probabilities = {}
        
        for model_name, model in self.models.items():
            predictions = []
            probabilities = []
            model.eval()
            
            with torch.no_grad():
                for images, _ in data_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
            
            model_predictions[model_name] = np.array(predictions)
            model_probabilities[model_name] = np.array(probabilities)
        
        return model_predictions, model_probabilities, true_labels
    
    def calculate_model_errors(self, model_predictions, true_labels):
        """
        計算每個模型的錯誤率
        
        Args:
            model_predictions (dict): 每個模型的預測結果
            true_labels (array): 真實標籤
            
        Returns:
            dict: 每個模型的錯誤率
        """
        model_errors = {}
        
        for model_name, predictions in model_predictions.items():
            error_rate = np.mean(predictions != true_labels)
            model_errors[model_name] = error_rate
        
        return model_errors
    
    def update_weights_boosting(self, model_errors):
        """
        基於 Boosting 策略更新模型權重
        
        Args:
            model_errors (dict): 每個模型的錯誤率
        """
        # 計算模型強度（錯誤率越低，強度越高）
        model_strengths = {}
        for model_name, error_rate in model_errors.items():
            # 避免除零錯誤
            error_rate = max(error_rate, 1e-10)
            error_rate = min(error_rate, 1 - 1e-10)
            
            # AdaBoost 權重公式：alpha = 0.5 * ln((1 - error) / error)
            alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            model_strengths[model_name] = alpha
        
        # 更新權重：使用學習率進行平滑更新
        for model_name in self.model_weights.keys():
            old_weight = self.model_weights[model_name]
            strength = model_strengths[model_name]
            
            # 平滑更新：new_weight = (1 - lr) * old_weight + lr * normalized_strength
            new_weight = old_weight + self.learning_rate * strength
            self.model_weights[model_name] = new_weight
        
        # 正規化權重（確保總和為1且所有權重為正）
        min_weight = min(self.model_weights.values())
        if min_weight < 0:
            # 如果有負權重，將所有權重向上平移
            shift = -min_weight + 0.01
            for model_name in self.model_weights.keys():
                self.model_weights[model_name] += shift
        
        # 正規化
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights.keys():
                self.model_weights[model_name] /= total_weight
        else:
            # 如果總權重為0，重置為平均權重
            for model_name in self.model_weights.keys():
                self.model_weights[model_name] = 1.0 / len(self.model_weights)
    
    def train_boosting_weights(self, weight_train_loader):
        """
        使用 Boosting 策略訓練模型權重
        
        Args:
            weight_train_loader: 權重訓練資料載入器
        """
        print(f"\n開始 Boosting 權重學習（{self.n_iterations} 輪迭代）...")
        
        for iteration in range(self.n_iterations):
            print(f"\n--- 第 {iteration + 1} 輪 ---")
            
            # 獲取當前權重下的預測結果
            model_predictions, model_probabilities, true_labels = self.get_model_predictions(weight_train_loader)
            
            # 計算每個模型的錯誤率
            model_errors = self.calculate_model_errors(model_predictions, true_labels)
            
            # 計算當前集成的準確率
            current_ensemble_pred = self.weighted_ensemble_predict(model_probabilities)
            current_accuracy = accuracy_score(true_labels, current_ensemble_pred)
            
            print(f"當前模型錯誤率: {model_errors}")
            print(f"當前權重: {self.model_weights}")
            print(f"當前集成準確率: {current_accuracy:.4f}")
            
            # 記錄歷史
            self.weight_history.append(self.model_weights.copy())
            self.error_history.append(current_accuracy)
            
            # 檢查收斂條件（如果準確率沒有顯著提升）
            if iteration > 0 and abs(self.error_history[-1] - self.error_history[-2]) < 0.001:
                print(f"在第 {iteration + 1} 輪達到收斂，提前停止")
                break
            
            # 更新權重
            if iteration < self.n_iterations - 1:  # 最後一輪不更新權重
                self.update_weights_boosting(model_errors)
                print(f"更新後權重: {self.model_weights}")
        
        print(f"\nBoosting 權重學習完成")
        print(f"最終權重: {self.model_weights}")
        
        # 顯示權重變化歷史
        self._plot_weight_evolution()
        
        return self.model_weights
    
    def weighted_ensemble_predict(self, model_probabilities):
        """
        使用當前權重進行加權集成預測
        
        Args:
            model_probabilities (dict): 每個模型的預測機率
            
        Returns:
            array: 集成預測結果
        """
        weighted_probs = None
        
        for model_name, probs in model_probabilities.items():
            weight = self.model_weights[model_name]
            weighted_prob = probs * weight
            
            if weighted_probs is None:
                weighted_probs = weighted_prob
            else:
                weighted_probs += weighted_prob
        
        # 獲取預測類別
        predictions = np.argmax(weighted_probs, axis=1)
        return predictions
    
    def _plot_weight_evolution(self):
        """
        繪製權重變化歷史
        """
        if len(self.weight_history) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 繪製權重變化
        plt.subplot(2, 1, 1)
        iterations = range(1, len(self.weight_history) + 1)
        
        for model_name in self.model_weights.keys():
            weights = [w[model_name] for w in self.weight_history]
            plt.plot(iterations, weights, marker='o', label=model_name, linewidth=2)
        
        plt.title('Model Weight Evolution During Boosting', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 繪製準確率變化
        plt.subplot(2, 1, 2)
        plt.plot(iterations, self.error_history, marker='s', color='green', linewidth=2)
        plt.title('Ensemble Accuracy During Boosting', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖片
        try:
            save_dir = "D:/實驗結果"
            os.makedirs(save_dir, exist_ok=True)
            if not os.access(save_dir, os.W_OK):
                save_dir = os.getcwd()
            
            save_path = os.path.join(save_dir, f"boosting_weight_evolution_{time.strftime('%Y%m%d', time.localtime())}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"權重變化圖已儲存至: {save_path}")
        except Exception as e:
            print(f"儲存權重變化圖失敗: {str(e)}")
        
        plt.show()
    
    def boosting_predict(self, test_loader):
        """
        使用訓練好的權重進行 Boosting 預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            tuple: (predictions, true_labels, prediction_probabilities)
        """
        print("\n正在進行 Boosting 預測 (加權 Soft Voting)...")
        print(f"使用權重: {self.model_weights}")
        
        # 獲取模型預測
        model_predictions, model_probabilities, true_labels = self.get_model_predictions(test_loader)
        
        # 計算加權集成預測機率
        ensemble_probabilities = None
        
        for model_name, probs in model_probabilities.items():
            weight = self.model_weights[model_name]
            weighted_prob = probs * weight
            
            if ensemble_probabilities is None:
                ensemble_probabilities = weighted_prob
            else:
                ensemble_probabilities += weighted_prob
        
        # 獲取最終預測類別
        final_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        print("Boosting 預測完成")
        
        return final_predictions, true_labels, ensemble_probabilities
    
    def save_boosting_weights(self, save_path="boosting_weights.pkl"):
        """
        儲存訓練好的 Boosting 權重
        """
        try:
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else os.getcwd()
            os.makedirs(save_dir, exist_ok=True)
            
            weights_info = {
                'model_weights': self.model_weights,
                'weight_history': self.weight_history,
                'error_history': self.error_history,
                'class_names': self.class_names,
                'n_iterations': self.n_iterations,
                'learning_rate': self.learning_rate
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(weights_info, f)
            
            print(f"Boosting 權重已儲存至: {save_path}")
        except Exception as e:
            print(f"儲存 Boosting 權重失敗: {str(e)}")
    
    def load_boosting_weights(self, load_path="boosting_weights.pkl"):
        """
        載入預訓練的 Boosting 權重
        """
        try:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    weights_info = pickle.load(f)
                
                self.model_weights = weights_info['model_weights']
                self.weight_history = weights_info.get('weight_history', [])
                self.error_history = weights_info.get('error_history', [])
                if self.class_names is None:
                    self.class_names = weights_info.get('class_names')
                
                print(f"Boosting 權重已從 {load_path} 載入")
                print(f"載入的權重: {self.model_weights}")
                return True
            else:
                print(f"找不到權重檔案: {load_path}")
                return False
        except Exception as e:
            print(f"載入 Boosting 權重失敗: {str(e)}")
            return False

    def export_misclassified_samples(self, test_loader, predicted_classes, true_labels, export_dir="D:/實驗結果/錯判樣本_Boosting"):
        """
        導出被錯誤分類的樣本圖片
        """
        print("\n正在導出錯誤分類的樣本圖片...")
        
        try:
            os.makedirs(export_dir, exist_ok=True)
            if not os.access(export_dir, os.W_OK):
                export_dir = os.path.join(os.getcwd(), "錯判樣本_Boosting")
                os.makedirs(export_dir, exist_ok=True)
                print(f"改用當前目錄: {export_dir}")
        except Exception as e:
            print(f"創建導出目錄失敗: {str(e)}")
            return 0
        
        dataset = test_loader.dataset
        # 處理 Subset 的情況
        if hasattr(dataset, 'dataset'):
            original_dataset = dataset.dataset
            indices = dataset.indices
            samples = [original_dataset.samples[i] for i in indices]
        else:
            samples = dataset.samples
        
        misclassified_indices = np.where(predicted_classes != true_labels)[0]
        total_misclassified = len(misclassified_indices)
        
        if total_misclassified == 0:
            print("沒有錯誤分類的樣本！")
            return 0
        
        print(f"發現 {total_misclassified} 個錯誤分類的樣本")
        
        import shutil
        
        for idx in misclassified_indices:
            try:
                img_path, _ = samples[idx]
                true_label = true_labels[idx]
                pred_label = predicted_classes[idx]
                
                true_class_name = self.class_names[true_label]
                pred_class_name = self.class_names[pred_label]
                
                error_class_dir = os.path.join(export_dir, f"{true_class_name}_誤判為_{pred_class_name}")
                os.makedirs(error_class_dir, exist_ok=True)
                
                filename = os.path.basename(img_path)
                dest_path = os.path.join(error_class_dir, filename)
                
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
        print("Boosting 模型評估結果")
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
        plt.title('Confusion Matrix - Ensemble Model (Boosting)')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"混淆矩陣已儲存至: {save_path}")
            except Exception as e:
                print(f"儲存混淆矩陣失敗: {str(e)}")
        
        plt.show()
        
        return cm
    
    def save_ensemble_model(self, save_dir="D:/模型儲存/ensemble_models"):
        """
        儲存完整的 Boosting 集成模型
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if not os.access(save_dir, os.W_OK):
                save_dir = os.path.join(os.getcwd(), "ensemble_models")
                os.makedirs(save_dir, exist_ok=True)
                print(f"改用當前目錄: {save_dir}")
            
            # 儲存完整的 Boosting 配置
            boosting_config = {
                'model_type': 'boosting',
                'model_weights': self.model_weights,
                'weight_history': self.weight_history,
                'error_history': self.error_history,
                'class_names': self.class_names,
                'model_paths': self.model_paths,
                'n_iterations': self.n_iterations,
                'learning_rate': self.learning_rate,
                'weight_train_ratio': self.weight_train_ratio,
                'device': str(self.device),
                'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())
            }
            
            config_path = os.path.join(save_dir, "boosting_ensemble_complete.pkl")
            with open(config_path, 'wb') as f:
                pickle.dump(boosting_config, f)
            
            print(f"完整 Boosting 集成模型已儲存至: {config_path}")
            
            # 同時儲存權重到原有格式（向後相容）
            weights_path = os.path.join(save_dir, f"boosting_weights_{time.strftime('%Y%m%d', time.localtime())}.pkl")
            self.save_boosting_weights(weights_path)
            
            return save_dir
            
        except Exception as e:
            print(f"儲存 Boosting 集成模型失敗: {str(e)}")
            return None
    
    def load_ensemble_model(self, load_dir="D:/模型儲存/ensemble_models"):
        """
        載入完整的 Boosting 集成模型
        """
        try:
            config_path = os.path.join(load_dir, "boosting_ensemble_complete.pkl")
            
            if not os.path.exists(config_path):
                # 嘗試載入舊格式的權重檔案
                weights_files = [f for f in os.listdir(load_dir) if f.startswith("boosting_weights_") and f.endswith(".pkl")]
                if weights_files:
                    weights_path = os.path.join(load_dir, weights_files[-1])  # 取最新的
                    return self.load_boosting_weights(weights_path)
                else:
                    print(f"找不到 Boosting 集成模型檔案: {config_path}")
                    return False
            
            with open(config_path, 'rb') as f:
                boosting_config = pickle.load(f)
            
            # 載入所有配置
            self.model_weights = boosting_config['model_weights']
            self.weight_history = boosting_config.get('weight_history', [])
            self.error_history = boosting_config.get('error_history', [])
            self.class_names = boosting_config['class_names']
            self.n_iterations = boosting_config.get('n_iterations', self.n_iterations)
            self.learning_rate = boosting_config.get('learning_rate', self.learning_rate)
            
            print(f"完整 Boosting 集成模型已載入: {config_path}")
            print(f"載入的最終權重: {self.model_weights}")
            print(f"訓練歷史: {len(self.weight_history)} 輪迭代")
            
            return True
            
        except Exception as e:
            print(f"載入 Boosting 集成模型失敗: {str(e)}")
            return False

    def run_ensemble_evaluation(self):
        """
        執行完整的 Boosting 集成評估流程
        """
        try:
            # 1. 載入模型
            self.load_models()
            
            # 2. 載入資料
            weight_train_loader, final_test_loader = self.load_data()
            
            # 3. 使用 Boosting 訓練權重
            final_weights = self.train_boosting_weights(weight_train_loader)
            
            # 4. 儲存權重
            try:
                save_dir = f"D:/模型儲存/ensemble_boosting_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
                saved_path = self.save_ensemble_model(save_dir)
                if saved_path:
                    print(f"完整 Boosting 集成模型已儲存至: {saved_path}")
            except Exception as save_error:
                print(f"儲存集成模型時發生錯誤: {str(save_error)}")
            
            # 5. 進行 Boosting 預測
            predicted_classes, true_labels, prediction_probs = self.boosting_predict(final_test_loader)
            
            # 6. 評估性能
            accuracy, report = self.evaluate_performance(predicted_classes, true_labels)
            
            # 7. 繪製混淆矩陣
            try:
                confusion_matrix_dir = "D:/實驗結果"
                os.makedirs(confusion_matrix_dir, exist_ok=True)
                if not os.access(confusion_matrix_dir, os.W_OK):
                    confusion_matrix_dir = os.getcwd()
                
                confusion_matrix_path = os.path.join(confusion_matrix_dir, 
                    f"confusion_matrix_boosting_{time.strftime('%Y%m%d', time.localtime())}.png")
                cm = self.plot_confusion_matrix(true_labels, predicted_classes, confusion_matrix_path)
            except Exception as cm_error:
                print(f"儲存混淆矩陣時發生錯誤: {str(cm_error)}")
                cm = self.plot_confusion_matrix(true_labels, predicted_classes, None)
            
            # 8. 導出錯誤分類的樣本
            export_dir = f"D:/實驗結果/錯判樣本_Boosting_{time.strftime('%Y%m%d', time.localtime())}"
            misclassified_count = self.export_misclassified_samples(final_test_loader, predicted_classes, true_labels, export_dir)
            
            print("\n" + "="*50)
            print("Boosting 集成評估完成！")
            print("="*50)
            print(f"最終權重: {final_weights}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': predicted_classes,
                'true_labels': true_labels,
                'prediction_probabilities': prediction_probs,
                'final_weights': final_weights,
                'weight_history': self.weight_history,
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
    print("Boosting Ensemble 模型評估系統 (PyTorch)")
    print("="*50)
    
    # 定義模型路徑
    model_paths = {
        'ResNet34': 'D:/模型儲存/ResNet34_model.pth',
        'AlexNet': 'D:/模型儲存/AlexNet_model.pth',
        'VGG16': 'D:/模型儲存/VGG16_model.pth'
    }
    
    # 定義測試資料路徑
    test_data_path = 'D:/資料集/root'
    
    # 創建 Boosting 集成分類器
    boosting_classifier = BoostingEnsembleClassifier(
        model_paths=model_paths,
        test_data_path=test_data_path,
        weight_train_ratio=0.3,  # 30% 用於權重學習
        n_iterations=5,          # 5輪 Boosting 迭代
        learning_rate=0.1        # 權重更新學習率
    )
    
    # 執行評估
    results = boosting_classifier.run_ensemble_evaluation()
    
    if results:
        print(f"\n最終準確率: {results['accuracy']:.4f}")
        print(f"最終權重: {results['final_weights']}")
        print(f"錯誤分類樣本數: {results['misclassified_count']}")
    else:
        print("評估失敗，請檢查模型和資料路徑是否正確")

if __name__ == "__main__":
    main()
