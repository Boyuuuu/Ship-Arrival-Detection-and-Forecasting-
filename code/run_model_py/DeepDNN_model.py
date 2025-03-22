import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据处理器（修复初始化问题）
class DataProcessor:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.filter(regex='^IP_').values.astype(np.float32)
        y = data.filter(regex='^OP_EP_').values.astype(np.float32)
        return X, y
    
    def fit(self, X_train, y_train):
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train)
        
    def transform(self, X, y=None):
        X_scaled = self.scaler_X.transform(X)
        if y is not None:
            y_scaled = self.scaler_y.transform(y)
            return X_scaled, y_scaled
        return X_scaled
    
    def inverse_transform_y(self, y):
        return self.scaler_y.inverse_transform(y)

# 数据集类（优化内存管理）
class TabularDataset(Dataset):
    def __init__(self, X, y, processor: DataProcessor):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].to(device), self.y[idx].to(device)

# 深度残差网络（修复初始化问题）
class DeepResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            self._make_res_block(256),
            self._make_res_block(256),
        )
        
        self.final = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_size)
        )
        
        # 修正初始化方法
        self._init_weights()
    
    def _make_res_block(self, dim):
        return nn.Sequential(
            ResidualLayer(dim),
            ResidualLayer(dim)
        )
    
    def _init_weights(self):
        """使用兼容GELU的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.normal_(m.bias, mean=0, std=0.01)
    
    def forward(self, x):
        x = self.main(x)
        return self.final(x)

class ResidualLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

# 训练器（优化日志记录）
class RegressionTrainer:
    def __init__(self, model, optimizer, scheduler, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, criterion):
        self.model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = self.model(X_val)
                val_loss += criterion(outputs, y_val).item()
        return val_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, criterion, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 动态早停机制
            if val_loss < self.best_loss * 0.995:
                self.best_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.plot_losses()
        
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title("Training Progress")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')
        plt.show()

# 主函数（优化流程控制）
def main():
    # 配置参数
    config = {
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 300,
        'patience': 15
    }
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据
    X_train, y_train = processor.load_data(r'Data\split_datasets\train.csv')
    X_val, y_val = processor.load_data(r'Data\split_datasets\val.csv')
    X_test, y_test = processor.load_data(r'Data\split_datasets\test.csv')
    
    # 标准化处理
    processor.fit(X_train, y_train)
    X_train, y_train = processor.transform(X_train, y_train)
    X_val, y_val = processor.transform(X_val, y_val)
    X_test, y_test = processor.transform(X_test, y_test)
    
    # 创建数据集
    train_set = TabularDataset(X_train, y_train, processor)
    val_set = TabularDataset(X_val, y_val, processor)
    test_set = TabularDataset(X_test, y_test, processor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'])
    
    # 初始化模型
    model = DeepResNet(input_size=X_train.shape[1], output_size=y_train.shape[1]).to(device)
    print(f"Model architecture:\n{model}")
    
    # 定义优化组件
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    # 训练模型
    trainer = RegressionTrainer(model, optimizer, scheduler, config['patience'])
    trainer.fit(train_loader, val_loader, criterion, config['epochs'])
    
    # 最终评估
    test_loss = trainer.validate(test_loader, criterion)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # 示例预测
    with torch.no_grad():
        sample_X, sample_y = test_set[0]
        prediction = model(sample_X.unsqueeze(0).to(device))
        true_value = processor.inverse_transform_y(sample_y.cpu().numpy().reshape(1, -1))
        pred_value = processor.inverse_transform_y(prediction.cpu().numpy())
        print(f"\nSample Prediction:\nTrue: {true_value[0]}\nPred: {pred_value[0]}")

if __name__ == "__main__":
    main()