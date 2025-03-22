# runmodel_pridict_city.py 最终修正版
import os
import torch
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from pridict_city_model_train import PortPredictor, FeatureEngineer, CONFIG
from torch.utils.data import Dataset, DataLoader

class PortDataset(Dataset):
    def __init__(self, data_path: str, engineer: FeatureEngineer = None):
        # 读取数据
        raw_df = pd.read_csv(data_path)
        raw_df = raw_df.dropna().reset_index(drop=True)  # 数据清理
        
        # 初始化特征工程对象
        self.engineer = engineer or FeatureEngineer()
        if engineer is None:
            self.engineer.fit(raw_df)  # 训练模式下拟合预处理器
        
        # 处理目标列（如果存在）
        target_cols = raw_df.filter(regex='OP_OH_EPT').columns
        if len(target_cols) > 0:
            y = raw_df[target_cols].values.argmax(axis=1)
            if engineer is None:
                # 训练模式下计算有效类别
                valid_counts = pd.Series(y).value_counts()
                self.engineer.valid_classes = valid_counts[valid_counts >= 2].index
                mask = pd.Series(y).isin(self.engineer.valid_classes)
                self.df = raw_df[mask].reset_index(drop=True)
            else:
                # 预测模式下过滤类别
                mask = pd.Series(y).isin(engineer.valid_classes)
                self.df = raw_df[mask].reset_index(drop=True)
        else:
            # 如果没有目标列，直接使用全部数据
            self.df = raw_df
        
        # 特征转换
        self.features = torch.as_tensor(self.engineer.transform(self.df))
        print(f"输入层的变量数量: {self.features.shape[1]}")
        
        # 处理标签（如果存在目标列）
        if len(target_cols) > 0:
            self.labels = self._process_labels()
            self.num_classes = len(self.engineer.valid_classes)
        else:
            self.labels = None
            self.num_classes = 0

    def _process_labels(self) -> torch.Tensor:
        target_cols = self.df.filter(regex='OP_OH_EPT').columns
        y = self.df[target_cols].values.argmax(axis=1)
        label_map = {old: new for new, old in enumerate(sorted(self.engineer.valid_classes))}
        return torch.tensor(np.vectorize(label_map.get)(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

def predict(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = PortPredictor(input_dim=183, output_dim=87)
    checkpoint = torch.load(config['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # 2. 加载预处理器
    engineer = FeatureEngineer.load(config['preprocessor_path'])
    
    # 3. 加载数据（原始数据路径需与主函数一致）
    dataset = PortDataset(
        data_path=r"Data\split_datasets\One_Hot_dataset\pridict_port\all.csv",
        engineer=engineer
    )
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    # 4. 执行预测
    predictions = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    # 5. 转换类别标签（如果有标签）
    if dataset.labels is not None:
        label_mapping = {idx: cls for idx, cls in enumerate(sorted(dataset.engineer.valid_classes))}
        final_predictions = [label_mapping[p] for p in predictions]
    else:
        final_predictions = predictions
    
    # 6. 将预测结果添加到处理后的数据框
    df_processed = dataset.df.copy()
    df_processed['Predicted_Class'] = final_predictions
    
    return df_processed

if __name__ == "__main__":
    # 配置参数（需要与训练时一致）
    CONFIG = {
        "model_save_path": r"Result\Predict_city\best_model.pth",
        "preprocessor_path": r"Result\Predict_city\preprocessor.joblib"
    }
    
    # 执行预测并获取处理后的数据和预测结果
    input_data_path = r"Data\split_datasets\One_Hot_dataset\pridict_port\all.csv"
    output_data_path = r"Data\split_datasets\One_Hot_dataset\pridict_port\all_with_predictions.csv"
    
    # 执行预测，返回处理后的数据框和预测结果
    df_processed = predict(CONFIG)
    
    # 保存合并后的数据到新文件
    df_processed.to_csv(output_data_path, index=False)
    print(f"合并后的数据已保存到 {output_data_path}")
    print(f"预测结果样例：{df_processed['Predicted_Class'].head(10).tolist()}")