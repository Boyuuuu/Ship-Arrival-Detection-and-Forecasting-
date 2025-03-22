import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    "data_folder": r"Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\pridict_Stime",
    "model_save_path": r"Result\Predict_Stime\best_model.pth",
    "preprocessor_path": r"Result\Predict_Stime\preprocessor.joblib",
    "batch_size": 4096,
    "max_epochs": 800,
    "patience": 20,
    "metrics_plot_path": r"Result\Predict_Stime\training_metrics.png",
    "confusion_matrix_path": r"Result\Predict_Stime\confusion_matrix.png",
    "prf_curve_path": r"Result\Predict_Stime\prf_curves.png" , # 新增PRF曲线路径
    "index_table_path": r"Result\Predict_Stime\index_mapping.csv",
    "metrics_plot_path": r"Result\Predict_Stime\loss_curves.png",
    "prf_curve_path": r"Result\Predict_Stime\true_vs_predicted.png"
}

class FeatureEngineer:
    def __init__(self):
        self.num_transformer = ColumnTransformer([
            ('log_dw', PowerTransformer(), ['IP_VDW']),
            ('sqrt_len', PowerTransformer(method='yeo-johnson'), ['IP_VL']),
            ('scale', StandardScaler(), ['IP_VBY','IP_VW','IP_VH','IP_VD','IP_VMS','IP_SPTO','IP_EPTO'])
        ])
        self.geo_transformer = ColumnTransformer([
            ('quantile', QuantileTransformer(n_quantiles=1000), ['IP_SPLO', 'IP_SPLA','IP_EPLO','IP_EPLA'])
        ])
        self.target_scaler = StandardScaler()  # 新增目标变量标准化器

    def _handle_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['IP_VDW'] = np.clip(df['IP_VDW'], 1, None)
        df['IP_VDW'] = np.log1p(df['IP_VDW'])
        df['IP_VL'] = np.sqrt(np.clip(df['IP_VL'], 0, None))
        return df
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        df = self._handle_skewness(df)
        self.num_transformer.fit(df)
        self.geo_transformer.fit(df[['IP_SPLO', 'IP_SPLA','IP_EPLO','IP_EPLA']])
        # 新增目标变量拟合
        self.target_scaler.fit(df[['OP_Stime']])
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        df = self._handle_skewness(df)
        num_features = self.num_transformer.transform(df)
        geo_features = self.geo_transformer.transform(df[['IP_SPLO', 'IP_SPLA','IP_EPLO','IP_EPLA']])
        cat_features = df.filter(regex='IP_OH_VST|IP_OH_VT').values
        
        processed = np.hstack([num_features, geo_features, cat_features])
        if np.isnan(processed).any() or np.isinf(processed).any():
            raise ValueError("特征数据包含无效值")
        return processed.astype(np.float32)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """转换目标变量"""
        return self.target_scaler.transform(y.reshape(-1, 1)).flatten()

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """逆转换目标变量"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def save(self, path: str) -> None:
        joblib.dump({
            'num_transformer': self.num_transformer,
            'geo_transformer': self.geo_transformer,
            'target_scaler': self.target_scaler  # 保存目标scaler
        }, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        data = joblib.load(path)
        engineer = cls()
        engineer.num_transformer = data['num_transformer']
        engineer.geo_transformer = data['geo_transformer']
        engineer.target_scaler = data['target_scaler']
        return engineer


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
        
        # 处理特征
        if engineer is None:
            self.engineer.fit(raw_df)  # 训练模式时拟合预处理器
        
        self.features = torch.as_tensor(self.engineer.transform_features(raw_df))
        
        # 处理目标变量
        y = raw_df['OP_Stime'].values.astype(np.float32)
        self.targets = torch.as_tensor(
            self.engineer.transform_target(y), 
            dtype=torch.float32
        )
        
        print(f"输入层的变量数量: {self.features.shape[1]}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.targets[idx]


class PortPredictor(nn.Module):
    def __init__(self, input_dim: int):
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
            ResidualBlock(512, 1024, 0.5),
            ResidualBlock(1024, 512, 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 输出层改为1个神经元
        )
        self._init_weights()
     

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # 去除多余的维度



def plot_metrics(history: Dict[str, list], config: Dict[str, Any]) -> None:
    plt.figure(figsize=(18, 6))
    
    # 训练/验证损失
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # 验证MAE
    plt.subplot(1, 3, 2)
    plt.plot(history['val_mae'], color='orange')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    # 新增：Custom Metric Sum 和 Avg
    plt.subplot(1, 3, 3)
    plt.plot(history['custom_metric_sum'], label='Custom Metric Sum', color='green')
    plt.plot(history['custom_metric_avg'], label='Custom Metric Avg', color='blue')
    plt.title('Custom Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
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



def generate_regression_report(model: nn.Module, dataloader: DataLoader, device: torch.device, engineer: FeatureEngineer):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            
            # 逆标准化得到原始尺度
            preds = engineer.inverse_transform_target(outputs)
            trues = engineer.inverse_transform_target(targets.numpy())
            
            y_pred.extend(preds)
            y_true.extend(trues)
    
    # 计算原有指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算新增指标
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # 公式: (168 - ((预测值 - 真实值)/3600)) / 168
    diff_hours = np.abs((y_pred_np - y_true_np) / 3600)  # 转换为小时差异，并取绝对值

    # 逻辑: 如果差异小于 168，则计算 custom_metric；否则设为 0
    custom_metric = np.where(
        diff_hours < 168,  # 条件
        (168.0 - diff_hours) / 168.0,  # 满足条件时的值
        0.0  # 不满足条件时的值
    )
    
    # 计算总和和平均值
    custom_metric_sum = np.sum(custom_metric)
    custom_metric_avg = np.mean(custom_metric)
    
    # 打印报告
    print(f"\nRegression Report:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Custom Metric Sum: {custom_metric_sum:.4f}")
    print(f"Custom Metric Avg: {custom_metric_avg:.4f}")
    
    # 绘制预测结果散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.savefig(CONFIG['prf_curve_path'])
    plt.close()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 初始化数据集
    train_set = PortDataset(os.path.join(CONFIG['data_folder'], 'train.csv'))
    val_set = PortDataset(os.path.join(CONFIG['data_folder'], 'val.csv'), train_set.engineer)
    
    # DataLoader
    train_loader = DataLoader( 
        train_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=12,  # 训练集设置为 12
        pin_memory=True,  # 启用 pin_memory
        persistent_workers=True  # 避免频繁重建 worker
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size'] * 2,
        num_workers=6,  # 验证集设置为 6
        pin_memory=True,  # 启用 pin_memory
        persistent_workers=True  # 避免频繁重建 worker
    )

    # 初始化模型
    model = PortPredictor(train_set.features.shape[1]).to(device)
    
    # 使用MSE损失函数
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'custom_metric_sum': [],  # 新增：记录 Custom Metric Sum
        'custom_metric_avg': []   # 新增：记录 Custom Metric Avg
    }
    best_loss = float('inf')

    for epoch in range(CONFIG['max_epochs']):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        custom_metric_sum = 0.0  # 新增：初始化 Custom Metric Sum
        custom_metric_avg = 0.0  # 新增：初始化 Custom Metric Avg
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # 计算验证损失和 MAE
                val_loss += criterion(outputs, targets).item()
                val_mae += nn.L1Loss()(outputs, targets).item()
                
                # 计算新增指标
                preds = train_set.engineer.inverse_transform_target(outputs.cpu().numpy())
                trues = train_set.engineer.inverse_transform_target(targets.cpu().numpy())
                
                diff_hours = np.abs((preds - trues) / 3600 ) # 转换为小时差异
                custom_metric = np.where(
                    diff_hours < 168,  # 条件
                    (168.0 - diff_hours) / 168.0,  # 满足条件时的值
                    0.0  # 不满足条件时的值
                )
                
                custom_metric_sum += np.sum(custom_metric)
                custom_metric_avg += np.mean(custom_metric)

        # 记录指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        history['custom_metric_sum'].append(custom_metric_sum)  # 新增：记录 Custom Metric Sum
        history['custom_metric_avg'].append(custom_metric_avg)  # 新增：记录 Custom Metric Avg
        
        # 输出指标
        print(f"Epoch {epoch+1}/{CONFIG['max_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val MAE: {avg_val_mae:.4f} | "
              f"Custom Metric Sum: {custom_metric_sum:.4f} | "  # 新增：输出 Custom Metric Sum
              f"Custom Metric Avg: {custom_metric_avg:.4f}")   # 新增：输出 Custom Metric Avg

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'engineer': train_set.engineer
            }, CONFIG['model_save_path'])

    # 绘制训练曲线
    plot_metrics(history, CONFIG)
    # 生成回归报告
    generate_regression_report(model, val_loader, device, train_set.engineer)


if __name__ == "__main__":
    train()