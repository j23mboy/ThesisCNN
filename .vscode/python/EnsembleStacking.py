import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
# 將 LogisticRegression 改為 LightGBM
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle

class StackingEnsembleClassifier:
    """
    集成分類器：使用 Stacking 整合三個預訓練模型
    Level 0: ResNet34, AlexNet, VGG16
    Level 1: LightGBM (Meta-Learner)
    """
    
    def __init__(self, model_paths, test_data_path, meta_train_ratio=0.3, n_folds=5, validation_data_path=None, lgb_params=None):
        """
        初始化 Stacking 集成分類器
        
        Args:
            model_paths (dict): 包含三個模型路徑的字典
            test_data_path (str): 測試資料集路徑
            meta_train_ratio (float): 用於訓練 meta-learner 的測試資料比例
            n_folds (int): 交叉驗證折數
            validation_data_path (str): 額外驗證資料集路徑（可選）
            lgb_params (dict): LightGBM 參數設定（可選）
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        self.validation_data_path = validation_data_path
        self.meta_train_ratio = meta_train_ratio
        self.n_folds = n_folds
        self.models = {}
        self.meta_learner = None
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 設定 LightGBM 參數
        if lgb_params is None:
            self.lgb_params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                'random_state': 42
            }
        else:
            self.lgb_params = lgb_params
        
        print(f"使用設備: {self.device}")
        print(f"Stacking 設定:")
        print(f"  - Meta-learner: LightGBM")
        print(f"  - Meta-learner 訓練資料比例: {meta_train_ratio}")
        print(f"  - 交叉驗證折數: {n_folds}")
        print(f"  - 使用額外驗證資料: {'是' if validation_data_path else '否'}")
        print(f"  - LightGBM 參數: {self.lgb_params}")
        
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
        載入資料集
        
        Returns:
            tuple: (meta_train_loader, meta_test_loader) 或 (validation_loader, test_loader)
        """
        print("正在載入資料...")
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if self.validation_data_path and os.path.exists(self.validation_data_path):
            # 使用額外的驗證資料集
            print("使用額外的驗證資料集訓練 meta-learner")
            
            validation_dataset = datasets.ImageFolder(
                root=self.validation_data_path,
                transform=transform
            )
            test_dataset = datasets.ImageFolder(
                root=self.test_data_path,
                transform=transform
            )
            
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            print(f"驗證資料集樣本數: {len(validation_dataset)}")
            print(f"測試資料集樣本數: {len(test_dataset)}")
            
            return validation_loader, test_loader
        
        else:
            # 分割測試資料集
            print(f"從測試資料集分割 {self.meta_train_ratio:.0%} 用於訓練 meta-learner")
            
            test_dataset = datasets.ImageFolder(
                root=self.test_data_path,
                transform=transform
            )
            
            # 分層採樣分割資料
            labels = [test_dataset[i][1] for i in range(len(test_dataset))]
            indices = list(range(len(test_dataset)))
            
            from sklearn.model_selection import train_test_split
            meta_train_indices, meta_test_indices = train_test_split(
                indices,
                test_size=1-self.meta_train_ratio,
                stratify=labels,
                random_state=42
            )
            
            # 創建子集
            meta_train_subset = Subset(test_dataset, meta_train_indices)
            meta_test_subset = Subset(test_dataset, meta_test_indices)
            
            # 創建資料載入器
            meta_train_loader = DataLoader(
                meta_train_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            meta_test_loader = DataLoader(
                meta_test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            self.class_names = test_dataset.classes
            print(f"Meta-learner 訓練樣本數: {len(meta_train_subset)}")
            print(f"最終測試樣本數: {len(meta_test_subset)}")
            print(f"發現 {len(self.class_names)} 個類別: {self.class_names}")
            
            return meta_train_loader, meta_test_loader
    
    def extract_base_predictions(self, data_loader):
        """
        提取基礎模型的預測機率作為特徵
        
        Args:
            data_loader: 資料載入器
            
        Returns:
            tuple: (features, labels)
        """
        print("正在提取基礎模型的預測特徵...")
        
        # 收集真實標籤
        true_labels = []
        for _, labels in data_loader:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        
        # 收集各模型的預測機率
        all_features = []
        
        for model_name, model in self.models.items():
            print(f"  正在提取 {model_name} 的預測機率...")
            
            model_predictions = []
            model.eval()
            
            with torch.no_grad():
                for images, _ in data_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    model_predictions.extend(probabilities.cpu().numpy())
            
            model_predictions = np.array(model_predictions)
            all_features.append(model_predictions)
            print(f"    {model_name} 特徵形狀: {model_predictions.shape}")
        
        # 將所有模型的預測機率連接成特徵向量
        # 形狀: (n_samples, n_models * n_classes)
        features = np.concatenate(all_features, axis=1)
        
        print(f"合併後特徵形狀: {features.shape}")
        print(f"標籤形狀: {true_labels.shape}")
        
        return features, true_labels
    
    def train_meta_learner(self, meta_train_loader):
        """
        使用 5-fold 交叉驗證訓練 LightGBM meta-learner
        
        Args:
            meta_train_loader: meta-learner 訓練資料載入器
        """
        print(f"\n正在使用 {self.n_folds}-fold 交叉驗證訓練 LightGBM Meta-Learner...")
        
        # 提取特徵和標籤
        X_meta, y_meta = self.extract_base_predictions(meta_train_loader)
        
        # 設定 LightGBM 參數中的類別數
        num_classes = len(self.class_names)
        self.lgb_params['num_class'] = num_classes
        
        # 執行交叉驗證
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        print(f"開始 {self.n_folds}-fold 交叉驗證:")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_meta)):
            X_train_fold = X_meta[train_idx]
            X_val_fold = X_meta[val_idx]
            y_train_fold = y_meta[train_idx]
            y_val_fold = y_meta[val_idx]
            
            # 創建 LightGBM 資料集
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            # 訓練 LightGBM
            fold_meta_learner = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # 驗證
            val_pred_probs = fold_meta_learner.predict(X_val_fold, num_iteration=fold_meta_learner.best_iteration)
            val_pred = np.argmax(val_pred_probs, axis=1)
            fold_score = accuracy_score(y_val_fold, val_pred)
            cv_scores.append(fold_score)
            
            print(f"  Fold {fold+1}: 準確率 = {fold_score:.4f}")
        
        # 計算平均 CV 分數
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\n交叉驗證結果:")
        print(f"  平均準確率: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        
        # 使用所有 meta 訓練資料訓練最終的 meta-learner
        print("\n使用所有 meta 訓練資料訓練最終 LightGBM Meta-Learner...")
        
        # 創建完整訓練資料集
        train_data = lgb.Dataset(X_meta, label=y_meta)
        
        # 訓練最終模型
        self.meta_learner = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        print("LightGBM Meta-Learner 訓練完成")
        
        # 顯示特徵重要性
        self._print_feature_importance()
        
        return mean_cv_score, std_cv_score
    
    def _print_feature_importance(self):
        """
        顯示 LightGBM 模型的特徵重要性
        """
        if self.meta_learner is not None:
            print("\nLightGBM 特徵重要性:")
            print("-" * 30)
            
            importance = self.meta_learner.feature_importance()
            feature_names = []
            
            # 生成特徵名稱 (模型名稱 + 類別)
            model_names = list(self.models.keys())
            for model_name in model_names:
                for class_name in self.class_names:
                    feature_names.append(f"{model_name}_{class_name}")
            
            # 按重要性排序
            importance_pairs = list(zip(feature_names, importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 顯示前10個最重要的特徵
            for i, (feature, imp) in enumerate(importance_pairs[:10]):
                print(f"  {i+1:2d}. {feature:<20}: {imp:>6.0f}")
    
    def stacking_predict(self, test_loader):
        """
        使用 Stacking (LightGBM) 進行預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            tuple: (predictions, true_labels, prediction_probabilities)
        """
        print("\n正在進行 Stacking 預測 (使用 LightGBM)...")
        
        if self.meta_learner is None:
            raise ValueError("LightGBM Meta-learner 尚未訓練，請先調用 train_meta_learner()")
        
        # 提取測試資料的特徵
        X_test, y_test = self.extract_base_predictions(test_loader)
        
        # 使用 LightGBM meta-learner 進行預測
        print("使用 LightGBM Meta-Learner 進行最終預測...")
        
        # 獲取預測機率
        prediction_probabilities = self.meta_learner.predict(X_test, num_iteration=self.meta_learner.best_iteration)
        
        # 獲取預測類別
        predictions = np.argmax(prediction_probabilities, axis=1)
        
        print("Stacking 預測完成")
        
        return predictions, y_test, prediction_probabilities
    
    def save_meta_learner(self, save_path="D:/模型儲存/meta_learner_lightgbm_stacking.pkl"):
        """
        儲存訓練好的 LightGBM meta-learner
        """
        if self.meta_learner is not None:
            try:
                # 使用 os.path.join 確保路徑正確性
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                
                # 檢查目錄是否可寫入
                if not os.access(save_dir, os.W_OK):
                    # 如果無法寫入 D 槽，改用當前工作目錄
                    print(f"警告：無法寫入 {save_dir}，改用當前目錄")
                    save_path = os.path.join(os.getcwd(), "meta_learner_lightgbm_stacking.pkl")
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                
                # 使用正確的路徑分隔符
                model_save_path = save_path.replace('.pkl', '.txt')
                
                # 儲存 LightGBM 模型
                try:
                    self.meta_learner.save_model(model_save_path)
                    print(f"LightGBM Meta-learner 已儲存至: {model_save_path}")
                except Exception as e:
                    print(f"儲存 LightGBM 模型失敗: {str(e)}")
                    # 嘗試使用替代方法儲存
                    import joblib
                    model_save_path_alt = save_path.replace('.pkl', '.joblib')
                    joblib.dump(self.meta_learner, model_save_path_alt)
                    print(f"使用 joblib 儲存 LightGBM 模型至: {model_save_path_alt}")
                
                # 儲存其他必要資訊
                meta_info = {
                    'lgb_params': self.lgb_params,
                    'class_names': self.class_names,
                    'model_names': list(self.models.keys())
                }
                
                with open(save_path, 'wb') as f:
                    pickle.dump(meta_info, f)
                
                print(f"Meta 資訊已儲存至: {save_path}")
                
            except Exception as e:
                print(f"儲存 meta-learner 時發生錯誤: {str(e)}")
                print("嘗試使用當前目錄儲存...")
                
                # 最後備案：儲存到當前目錄
                try:
                    backup_path = os.path.join(os.getcwd(), "meta_learner_lightgbm_backup.pkl")
                    backup_model_path = backup_path.replace('.pkl', '.txt')
                    
                    self.meta_learner.save_model(backup_model_path)
                    
                    meta_info = {
                        'lgb_params': self.lgb_params,
                        'class_names': self.class_names,
                        'model_names': list(self.models.keys())
                    }
                    
                    with open(backup_path, 'wb') as f:
                        pickle.dump(meta_info, f)
                    
                    print(f"備用儲存成功: {backup_model_path}")
                    
                except Exception as backup_error:
                    print(f"備用儲存也失敗: {str(backup_error)}")
                    print("跳過模型儲存步驟，繼續執行評估...")
        else:
            print("沒有可儲存的 LightGBM meta-learner")
    
    def load_meta_learner(self, load_path="D:/模型儲存/meta_learner_lightgbm_stacking.pkl"):
        """
        載入預訓練的 LightGBM meta-learner
        """
        model_load_path = load_path.replace('.pkl', '.txt')
        
        # 檢查檔案是否存在
        if not os.path.exists(load_path):
            # 嘗試在當前目錄尋找備用檔案
            backup_path = os.path.join(os.getcwd(), "meta_learner_lightgbm_backup.pkl")
            backup_model_path = backup_path.replace('.pkl', '.txt')
            
            if os.path.exists(backup_path) and os.path.exists(backup_model_path):
                load_path = backup_path
                model_load_path = backup_model_path
                print("使用備用檔案載入 meta-learner")
            else:
                print(f"找不到 meta-learner 檔案: {load_path}")
                return False
        
        if not os.path.exists(model_load_path):
            print(f"找不到 LightGBM 模型檔案: {model_load_path}")
            return False
        
        try:
            # 載入 LightGBM 模型
            self.meta_learner = lgb.Booster(model_file=model_load_path)
            
            # 載入其他資訊
            with open(load_path, 'rb') as f:
                meta_info = pickle.load(f)
                self.lgb_params = meta_info['lgb_params']
                if self.class_names is None:
                    self.class_names = meta_info['class_names']
            
            print(f"LightGBM Meta-learner 已從 {model_load_path} 載入")
            return True
            
        except Exception as e:
            print(f"載入 meta-learner 失敗: {str(e)}")
            return False

    def export_misclassified_samples(self, test_loader, predicted_classes, true_labels, export_dir="D:/實驗結果/錯判樣本_Stacking"):
        """
        導出被錯誤分類的樣本圖片
        """
        print("\n正在導出錯誤分類的樣本圖片...")
        
        os.makedirs(export_dir, exist_ok=True)
        
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
        print("Stacking 模型評估結果 (LightGBM Meta-Learner)")
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
        plt.title('Confusion Matrix - Ensemble Model (Stacking with LightGBM)')
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
        儲存完整的 Stacking 集成模型
        
        Args:
            save_dir (str): 儲存目錄
            
        Returns:
            str: 儲存路徑，失敗時返回 None
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 檢查目錄是否可寫入
            if not os.access(save_dir, os.W_OK):
                save_dir = os.path.join(os.getcwd(), "ensemble_models")
                os.makedirs(save_dir, exist_ok=True)
                print(f"改用當前目錄: {save_dir}")
            
            # 儲存 LightGBM Meta-learner
            if self.meta_learner is not None:
                model_save_path = os.path.join(save_dir, "lightgbm_meta_learner.txt")
                try:
                    self.meta_learner.save_model(model_save_path)
                    print(f"LightGBM Meta-learner 已儲存至: {model_save_path}")
                except Exception as e:
                    print(f"儲存 LightGBM 模型失敗: {str(e)}")
                    # 使用 joblib 作為備用方案
                    import joblib
                    model_save_path_alt = os.path.join(save_dir, "lightgbm_meta_learner.joblib")
                    joblib.dump(self.meta_learner, model_save_path_alt)
                    print(f"使用 joblib 儲存 LightGBM 模型至: {model_save_path_alt}")
            
            # 儲存完整的 Stacking 配置
            stacking_config = {
                'model_type': 'stacking_lightgbm',
                'lgb_params': self.lgb_params,
                'meta_train_ratio': self.meta_train_ratio,
                'n_folds': self.n_folds,
                'class_names': self.class_names,
                'model_paths': self.model_paths,
                'validation_data_path': self.validation_data_path,
                'device': str(self.device),
                'timestamp': time.strftime('%Y%m%d_%H%M%S', time.localtime())
            }
            
            config_path = os.path.join(save_dir, "stacking_ensemble_complete.pkl")
            with open(config_path, 'wb') as f:
                pickle.dump(stacking_config, f)
            
            print(f"完整 Stacking 集成模型配置已儲存至: {config_path}")
            
            # 儲存基礎模型路徑資訊（用於重新載入）
            paths_info = {
                'base_models': self.model_paths,
                'ensemble_type': 'stacking_lightgbm',
                'meta_learner_type': 'lightgbm',
                'lgb_params': self.lgb_params
            }
            
            paths_path = os.path.join(save_dir, "model_paths_info.pkl")
            with open(paths_path, 'wb') as f:
                pickle.dump(paths_info, f)
            
            print(f"模型路徑資訊已儲存至: {paths_path}")
            
            # 同時使用舊格式儲存（向後相容）
            legacy_save_path = os.path.join(save_dir, "meta_learner_lightgbm_stacking.pkl")
            try:
                self.save_meta_learner(legacy_save_path)
            except Exception as legacy_error:
                print(f"舊格式儲存失敗: {str(legacy_error)}")
            
            return save_dir
            
        except Exception as e:
            print(f"儲存 Stacking 集成模型失敗: {str(e)}")
            return None
    
    def load_ensemble_model(self, load_dir="D:/模型儲存/ensemble_models"):
        """
        載入完整的 Stacking 集成模型
        
        Args:
            load_dir (str): 載入目錄
            
        Returns:
            bool: 載入是否成功
        """
        try:
            config_path = os.path.join(load_dir, "stacking_ensemble_complete.pkl")
            
            if not os.path.exists(config_path):
                # 嘗試載入舊格式
                legacy_path = os.path.join(load_dir, "meta_learner_lightgbm_stacking.pkl")
                if os.path.exists(legacy_path):
                    print("使用舊格式載入 Stacking 模型")
                    return self.load_meta_learner(legacy_path)
                else:
                    # 嘗試在當前目錄尋找
                    backup_dir = os.path.join(os.getcwd(), "ensemble_models")
                    config_path = os.path.join(backup_dir, "stacking_ensemble_complete.pkl")
                    
                    if not os.path.exists(config_path):
                        print(f"找不到 Stacking 集成模型檔案: {config_path}")
                        return False
            
            # 載入配置
            with open(config_path, 'rb') as f:
                stacking_config = pickle.load(f)
            
            # 載入配置參數
            self.lgb_params = stacking_config['lgb_params']
            self.meta_train_ratio = stacking_config.get('meta_train_ratio', self.meta_train_ratio)
            self.n_folds = stacking_config.get('n_folds', self.n_folds)
            self.class_names = stacking_config['class_names']
            self.validation_data_path = stacking_config.get('validation_data_path')
            
            print(f"Stacking 配置已載入: {config_path}")
            print(f"Meta-learner 類型: LightGBM")
            print(f"類別數量: {len(self.class_names)}")
            print(f"交叉驗證折數: {self.n_folds}")
            
            # 載入 LightGBM Meta-learner
            model_load_path = os.path.join(load_dir, "lightgbm_meta_learner.txt")
            model_load_path_alt = os.path.join(load_dir, "lightgbm_meta_learner.joblib")
            
            if os.path.exists(model_load_path):
                try:
                    import lightgbm as lgb
                    self.meta_learner = lgb.Booster(model_file=model_load_path)
                    print(f"LightGBM Meta-learner 已載入: {model_load_path}")
                except Exception as e:
                    print(f"載入 LightGBM 模型失敗: {str(e)}")
                    return False
            elif os.path.exists(model_load_path_alt):
                try:
                    import joblib
                    self.meta_learner = joblib.load(model_load_path_alt)
                    print(f"LightGBM Meta-learner 已從 joblib 載入: {model_load_path_alt}")
                except Exception as e:
                    print(f"從 joblib 載入模型失敗: {str(e)}")
                    return False
            else:
                print(f"找不到 LightGBM 模型檔案")
                return False
            
            return True
            
        except Exception as e:
            print(f"載入 Stacking 集成模型失敗: {str(e)}")
            return False
    
    def predict_with_saved_model(self, test_loader):
        """
        使用已儲存的 Stacking 集成模型進行預測
        
        Args:
            test_loader: 測試資料載入器
            
        Returns:
            tuple: 預測結果
        """
        if not self.models:
            print("基礎模型尚未載入，正在載入...")
            self.load_models()
        
        if self.meta_learner is None:
            print("Meta-learner 尚未載入，請先載入集成模型")
            return None
        
        # 使用現有的預測方法
        return self.stacking_predict(test_loader)

    def run_ensemble_evaluation(self):
        """
        執行完整的 Stacking 集成評估流程
        """
        try:
            # 1. 載入模型
            self.load_models()
            
            # 2. 載入資料
            meta_train_loader, test_loader = self.load_data()
            
            # 3. 訓練 LightGBM meta-learner
            cv_score, cv_std = self.train_meta_learner(meta_train_loader)
            
            # 4. 儲存完整的集成模型（新增）
            try:
                save_dir = f"D:/模型儲存/ensemble_stacking_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
                saved_path = self.save_ensemble_model(save_dir)
                if saved_path:
                    print(f"完整 Stacking 集成模型已儲存至: {saved_path}")
            except Exception as save_error:
                print(f"儲存集成模型時發生錯誤: {str(save_error)}")
                # 嘗試使用舊格式儲存作為備用
                try:
                    self.save_meta_learner()
                except Exception as legacy_error:
                    print(f"備用儲存也失敗: {str(legacy_error)}")
                    print("繼續執行評估流程...")
            
            # 5. 進行 Stacking 預測
            predicted_classes, true_labels, prediction_probs = self.stacking_predict(test_loader)
            
            # 6. 評估性能
            accuracy, report = self.evaluate_performance(predicted_classes, true_labels)
            
            # 7. 繪製混淆矩陣（修改儲存路徑）
            try:
                confusion_matrix_dir = "D:/實驗結果"
                os.makedirs(confusion_matrix_dir, exist_ok=True)
                if not os.access(confusion_matrix_dir, os.W_OK):
                    confusion_matrix_dir = os.getcwd()
                    print(f"無法寫入 D:/實驗結果，改用當前目錄: {confusion_matrix_dir}")
                
                confusion_matrix_path = os.path.join(confusion_matrix_dir, 
                    f"confusion_matrix_stacking_lightgbm_{time.strftime('%Y%m%d', time.localtime())}.png")
                cm = self.plot_confusion_matrix(true_labels, predicted_classes, confusion_matrix_path)
            except Exception as cm_error:
                print(f"儲存混淆矩陣時發生錯誤: {str(cm_error)}")
                cm = self.plot_confusion_matrix(true_labels, predicted_classes, None)
            
            # 8. 導出錯誤分類的樣本（修改導出路徑）
            try:
                export_base_dir = "D:/實驗結果"
                os.makedirs(export_base_dir, exist_ok=True)
                if not os.access(export_base_dir, os.W_OK):
                    export_base_dir = os.getcwd()
                    print(f"無法寫入 D:/實驗結果，改用當前目錄: {export_base_dir}")
                
                export_dir = os.path.join(export_base_dir, 
                    f"錯判樣本_Stacking_LightGBM_{time.strftime('%Y%m%d', time.localtime())}")
                misclassified_count = self.export_misclassified_samples(test_loader, predicted_classes, true_labels, export_dir)
            except Exception as export_error:
                print(f"導出錯判樣本時發生錯誤: {str(export_error)}")
                misclassified_count = len(np.where(predicted_classes != true_labels)[0])
            
            print("\n" + "="*50)
            print("Stacking 集成評估完成！(LightGBM Meta-Learner)")
            print("="*50)
            print(f"交叉驗證分數: {cv_score:.4f} ± {cv_std:.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': predicted_classes,
                'true_labels': true_labels,
                'prediction_probabilities': prediction_probs,
                'cv_score': cv_score,
                'cv_std': cv_std,
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
    print("Stacking Ensemble 模型評估系統 (PyTorch + LightGBM)")
    print("="*50)
    
    # 定義模型路徑
    model_paths = {
        'ResNet34': 'D:/模型儲存/ResNet34_model.pth',
        'AlexNet': 'D:/模型儲存/AlexNet_model.pth',
        'VGG16': 'D:/模型儲存/VGG16_model.pth'
    }
    
    # 定義測試資料路徑
    test_data_path = 'D:/資料集/root'
    
    # 可選：定義額外的驗證資料路徑
    # validation_data_path = 'D:/資料集/validation'  # 如果有的話
    validation_data_path = None
    
    # 自訂 LightGBM 參數 (可選)
    custom_lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    # 創建 Stacking 集成分類器
    stacking_classifier = StackingEnsembleClassifier(
        model_paths=model_paths,
        test_data_path=test_data_path,
        meta_train_ratio=0.3,  # 30% 用於訓練 meta-learner
        n_folds=5,  # 5-fold 交叉驗證
        validation_data_path=validation_data_path,
        lgb_params=custom_lgb_params  # 使用自訂 LightGBM 參數
    )
    
    # 執行評估
    results = stacking_classifier.run_ensemble_evaluation()
    
    if results:
        print(f"\n最終準確率: {results['accuracy']:.4f}")
        print(f"交叉驗證分數: {results['cv_score']:.4f} ± {results['cv_std']:.4f}")
        print(f"錯誤分類樣本數: {results['misclassified_count']}")
    else:
        print("評估失敗，請檢查模型和資料路徑是否正確")

if __name__ == "__main__":
    main()
