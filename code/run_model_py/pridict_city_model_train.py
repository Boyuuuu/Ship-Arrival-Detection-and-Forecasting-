import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "data_folder": r"Data/split_datasets/One_Hot_dataset",
    "model_save_path": r"Result\Predict_city\best_model.pth",
    "preprocessor_path": r"Result\Predict_city\preprocessor.joblib",
    "batch_size": 2048,
    "max_epochs": 800,
    "patience": 20,
    "metrics_plot_path": r"Result\Predict_city\training_metrics.png",
    "confusion_matrix_path": r"Result\Predict_city\confusion_matrix.png",
    "prf_curve_path": r"Result\Predict_city\prf_curves.png" , # 新增PRF曲线路径
    "index_table_path": r"Result\Predict_city\index_mapping.csv"
}

class FeatureEngineer:
    def __init__(self):
        self.num_transformer = ColumnTransformer([
            ('log_dw', PowerTransformer(), ['IP_VDW']),
            ('sqrt_len', PowerTransformer(method='yeo-johnson'), ['IP_VL']),
            ('scale', StandardScaler(), ['IP_VBY','IP_VW','IP_VH','IP_VD','IP_VMS'])
        ])
        self.geo_transformer = ColumnTransformer([
            ('quantile', QuantileTransformer(n_quantiles=1000), ['IP_SPLO', 'IP_SPLA'])
        ])
        self.valid_classes = None

    def _handle_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['IP_VDW'] = np.clip(df['IP_VDW'], 1, None)
        df['IP_VDW'] = np.log1p(df['IP_VDW'])
        df['IP_VL'] = np.sqrt(np.clip(df['IP_VL'], 0, None))
        return df

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        df = self._handle_skewness(df)
        self.num_transformer.fit(df)
        self.geo_transformer.fit(df[['IP_SPLO', 'IP_SPLA']])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._handle_skewness(df)
        num_features = self.num_transformer.transform(df)
        geo_features = self.geo_transformer.transform(df[['IP_SPLO', 'IP_SPLA']])
        cat_features = df.filter(regex='IP_OH_VST|IP_OH_VT|IP_OH_SPT').values
        
        processed = np.hstack([num_features, geo_features, cat_features])
        if np.isnan(processed).any() or np.isinf(processed).any():
            raise ValueError("特征数据包含无效值")
        return processed.astype(np.float32)

    def save(self, path: str) -> None:
        """保存预处理对象"""
        joblib.dump({
            'num_transformer': self.num_transformer,
            'geo_transformer': self.geo_transformer,
            'valid_classes': self.valid_classes
        }, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """加载预处理对象"""
        data = joblib.load(path)
        engineer = cls()
        engineer.num_transformer = data['num_transformer']
        engineer.geo_transformer = data['geo_transformer']
        engineer.valid_classes = data['valid_classes']
        return engineer

class PortDataset(Dataset):
    def __init__(self, data_path: str, engineer: FeatureEngineer = None):
        raw_df = pd.read_csv(data_path)
        self.engineer = engineer or FeatureEngineer()
        
        # 数据清理
        raw_df = raw_df.dropna().reset_index(drop=True)

        # 获取目标列
        target_cols = raw_df.filter(regex='OP_OH_EPT').columns  # 提前定义 target_cols
        y = raw_df[target_cols].values.argmax(axis=1)
        
        # 保存原始信息
        self.original_target_cols = target_cols  # 新增
        self.original_y = y   

        # 分类处理
        if engineer is None:
            # 内部计算有效类别并训练特征工程对象
            target_cols = raw_df.filter(regex='OP_OH_EPT').columns
            y = raw_df[target_cols].values.argmax(axis=1)
            valid_counts = pd.Series(y).value_counts()
            self.engineer.valid_classes = valid_counts[valid_counts >= 2].index
            mask = pd.Series(y).isin(self.engineer.valid_classes)
            self.df = raw_df[mask].reset_index(drop=True)
            self.engineer.fit(self.df)
        else:
            # 使用外部的 FeatureEngineer 对象
            target_cols = raw_df.filter(regex='OP_OH_EPT').columns
            y = raw_df[target_cols].values.argmax(axis=1)
            mask = pd.Series(y).isin(engineer.valid_classes)
            self.df = raw_df[mask].reset_index(drop=True)

                # 保存过滤掩码
        self.mask = mask  # 新增
        
        # 转换为Tensor并缓存
        self.features = torch.as_tensor(self.engineer.transform(self.df))
        print(f"输入层的变量数量: {self.features.shape[1]}")
        self.labels = self._process_labels()
        self.num_classes = len(self.engineer.valid_classes)

    def _process_labels(self) -> torch.Tensor:
        target_cols = self.df.filter(regex='OP_OH_EPT').columns
        y = self.df[target_cols].values.argmax(axis=1)
        label_map = {old: new for new, old in enumerate(sorted(self.engineer.valid_classes))}
        return torch.tensor(np.vectorize(label_map.get)(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx]

class PortPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

                # 定义残差块
        class ResidualBlock(nn.Module):
            def __init__(self, in_features: int, out_features: int, dropout_rate: float):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.GELU(),
                    nn.BatchNorm1d(out_features),
                    nn.Dropout(dropout_rate)
                )
                self.shortcut = nn.Sequential()
                if in_features != out_features:
                    self.shortcut = nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features)
                    )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.block(x) + self.shortcut(x)
        
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            ResidualBlock(512, 1024, 0.5),  # 残差块
            ResidualBlock(1024, 512, 0.5),  # 残差块
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def plot_metrics(history: Dict[str, list], config: Dict[str, Any]) -> None:
    """绘制训练指标曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(config['metrics_plot_path'])
    plt.close()

def plot_prf_curves(report_dict: dict, config: Dict[str, Any]) -> None:
    """绘制每个类别的 precision/recall/f1 曲线"""
    # 提取类别指标数据
    class_names = [k for k in report_dict.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    precisions = [report_dict[cls]['precision'] for cls in class_names]
    recalls = [report_dict[cls]['recall'] for cls in class_names]
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]

    plt.figure(figsize=(18, 6))
    
    # 绘制三条曲线
    plt.plot(class_names, precisions, label='Precision', marker='o', markersize=4, linestyle='-', linewidth=1, color='#1f77b4')
    plt.plot(class_names, recalls, label='Recall', marker='s', markersize=4, linestyle='--', linewidth=1, color='#ff7f0e')
    plt.plot(class_names, f1_scores, label='F1-Score', marker='^', markersize=4, linestyle='-.', linewidth=1, color='#2ca02c')
    
    # 优化显示效果
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision/Recall/F1-Score per Class', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 添加图例
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(config['prf_curve_path'], bbox_inches='tight')
    plt.close()

def generate_classification_report(model: nn.Module, dataloader: DataLoader, device: torch.device, class_names: list) -> None:
    """生成分类报告并绘制图表"""
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(targets.numpy())
            y_pred.extend(preds)
    
    # 确保使用训练集的完整类别标签
    labels = np.arange(len(class_names))  # 使用训练集的全部87个类别索引
    
    # 生成分类报告（显式指定labels参数）
    print("\nClassification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels,          # 强制包含所有87个类别
        target_names=class_names,
        zero_division=0
    ))
    
    # 生成混淆矩阵（同样指定labels）
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Including All 87 Classes)')
    plt.colorbar()
    
    # 设置刻度标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=4)
    plt.yticks(tick_marks, class_names, fontsize=4)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFIG['confusion_matrix_path'], dpi=300)
    plt.close()
    
    # 绘制PRF曲线（需新增此函数）
    plot_prf_curves(
        classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True),
        CONFIG
    )


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 初始化数据集
    train_set = PortDataset(os.path.join(CONFIG['data_folder'], 'train.csv'))
    val_set = PortDataset(os.path.join(CONFIG['data_folder'], 'val.csv'), train_set.engineer)
    
    print(f"训练集样本数: {len(train_set)} | 类别数: {train_set.num_classes}")
    print(f"验证集样本数: {len(val_set)} | 类别数: {val_set.num_classes}")


    target_cols = train_set.original_target_cols
    original_classes = np.arange(len(target_cols))
    valid_classes = train_set.engineer.valid_classes
    
    # 创建映射关系
    label_map = {old: new for new, old in enumerate(sorted(valid_classes))}
    
    # 统计有效数量
    valid_y = train_set.original_y[train_set.mask]
    valid_counts = pd.Series(valid_y).value_counts().reindex(original_classes, fill_value=0)
    
    # 构建DataFrame
    index_table = []
    for i, col in enumerate(target_cols):
        original_index = i
        if original_index in valid_classes:
            new_index = label_map[original_index]
            count = valid_counts.loc[original_index]
        else:
            new_index = -1
            count = 0
        index_table.append({
            'ori': col,
            'index': new_index,
            'valid': count
        })
    
    index_df = pd.DataFrame(index_table)
    
    # 保存文件
    index_df.to_csv(CONFIG['index_table_path'], index=False)
    print(f"索引表已保存至: {CONFIG['index_table_path']}")


    # DataLoader配置
    train_loader = DataLoader(
        train_set, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size']*2,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    model = PortPredictor(train_set.features.shape[1], train_set.num_classes).to(device)
    
    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4,
        total_steps=CONFIG['max_epochs']*len(train_loader),
        pct_start=0.3
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    best_acc = 0.0
    
    for epoch in range(CONFIG['max_epochs']):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                correct += (outputs.argmax(1) == targets).sum().item()
        
        # 记录指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_set)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['max_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, CONFIG['model_save_path'])
            train_set.engineer.save(CONFIG['preprocessor_path'])

    # 训练后处理
    plot_metrics(history, CONFIG)
    generate_classification_report(
        model, val_loader, device, 
        class_names=[str(cls) for cls in sorted(train_set.engineer.valid_classes)]
    )


if __name__ == "__main__":
    train()