<<<<<<< HEAD
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader
from torch.optim import optimizer

def generate_simulation_data(batch_size, window_size=5000, max_delay=100, max_points=1):
    """
    生成模拟的 MZ-Sagnac 多点重叠扰动数据及一维高斯热力图标签
    采样率参考 5 MSa/s [cite: 825]。
    """
    X = torch.zeros(batch_size, 2, window_size)
    # 标签热力图：长度为 2 * max_delay + 1 的一维数组
    Y_heatmap = torch.zeros(batch_size, 2 * max_delay + 1)

    t = np.arange(window_size)

    for b in range(batch_size):
        num_points = np.random.randint(1, max_points + 1)
        x_signal = np.zeros(window_size)
        y_signal = np.zeros(window_size)

        for p in range(num_points):
            # 模拟随机延迟和随机起振时间，制造严重的 Temporal Overlap
            delay = np.random.randint(-max_delay + 10, max_delay - 10)
            onset = np.random.randint(100, window_size - 2000)

            # 多频混合瞬态衰减信号
            A = np.random.uniform(0.5, 1.5)
            alpha = np.random.uniform(0.005, 0.01)
            freqs = [np.random.uniform(0.005,0.01), np.random.uniform(0.01,0.02)]

            # 构造 x(t)
            x_component = np.zeros(window_size)
            envelope = A * np.exp(-alpha * (t[onset:] - onset))
            for f in freqs:
                x_component[onset:] += envelope * np.sin(2 * np.pi * f * (t[onset:] - onset))

            # 构造 y(t) = x(t - delay)
            y_component = np.roll(x_component, delay)

            x_signal += x_component
            y_signal += y_component

            # 在对应延迟位置渲染一维高斯分布 (Sigma = 2.0)
            # 避免直接回归造成的偏移，利用高斯分布引导网络学习平滑特征
            center = delay + max_delay
            sigma = 2.0
            x_grid = np.arange(2 * max_delay + 1)
            gaussian = np.exp(-((x_grid - center) ** 2) / (2 * sigma ** 2))

            # 取多点高斯峰的最大值（如果峰重叠）
            # 这里认为当多个峰同时存在时，选择最大者，
            # 网络最终的输出要和这个热力图进行拟合
            Y_heatmap[b] = torch.max(Y_heatmap[b], torch.tensor(gaussian, dtype=torch.float32))

        # 添加高斯白噪声
        X[b, 0, :] = torch.tensor(x_signal) + torch.randn(window_size) * 0.001
        X[b, 1, :] = torch.tensor(y_signal) + torch.randn(window_size) * 0.001

    return X, Y_heatmap



class DelayEstimationDataset(Dataset):
    def __init__(self, num_samples, window_size, max_delay,max_points = 1, mode='train'):
        self.num_samples = num_samples
        self.window_size = window_size
        self.max_delay = max_delay
        self.max_points = max_points
        self.mode = mode

        # 预生成所有数据，避免训练时动态生成导致的IO瓶颈
        self.X,self.Y_heatmap = generate_simulation_data(num_samples, window_size, max_delay, max_points)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        val = self.X[idx]
        target = self.Y_heatmap[idx]
        return val,target


class NonCausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # 严格计算对称 Padding，确保非因果性（零相位偏移）
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,padding=pad,
                              padding_mode='replicate', dilation=dilation, bias=False)
        self.bn = nn.InstanceNorm1d(out_channels,affine=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FeatureCorrelationLayer(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

    def forward(self, f_x, f_y):
        """
        计算特征级别的互相关。
        输入维度: [B, C, L]
        输出维度: [B, C, 2*max_delay + 1]
        """
        B, C, L = f_x.size()
        correlations = []

        # 归一化特征，类似于原文公式(7)
        f_x = F.normalize(f_x, p=2, dim=2)
        f_y = F.normalize(f_y, p=2, dim=2)

        # 在特征维度进行不同 shift 的滑动点乘
        for shift in range(-self.max_delay, self.max_delay + 1):
            if shift < 0:
                x_shift = f_x[:, :, :shift]
                y_shift = f_y[:, :, -shift:]
            elif shift > 0:
                x_shift = f_x[:, :, shift:]
                y_shift = f_y[:, :, :-shift]
            else:
                x_shift = f_x
                y_shift = f_y

            # 计算相似度并对时间维度求和
            corr = torch.sum(x_shift * y_shift, dim=2, keepdim=True)  # [B, C, 1]
            correlations.append(corr)

        return torch.cat(correlations, dim=2)  # [B, C, 2*max_delay + 1]


class OptimizedFeatureCorrelationLayer(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

    def forward(self, f_x, f_y):
        """
        利用 F.conv1d 和 Grouped Convolution 极致优化计算图的互相关
        输入维度: f_x, f_y -> [B, C, L]
        输出维度: [B, C, 2*max_delay + 1]
        """
        B, C, L = f_x.size()

        f_x = f_x - f_x.mean(dim=-1, keepdim=True)
        f_y = f_y - f_y.mean(dim=-1, keepdim=True)

        # 1. 归一化 (计算 Cosine 相似度必备)
        f_x = F.normalize(f_x, p=2, dim=2)
        f_y = F.normalize(f_y, p=2, dim=2)

        # 2. 维度重构技巧
        # 互相关是在 x 上滑动 y。为了用 F.conv1d 批量处理，
        # 我们把 f_x 作为 "Input"，把 f_y 作为 "Weight"。
        # 为了不让不同 Batch 和不同 Channel 相互干扰，我们使用 分组卷积 (groups = B * C)

        # 将 f_x 展平为单 Batch，包含 B*C 个独立的组
        # 维度变换: [B, C, L] -> [1, B*C, L]
        f_x_reshaped = f_x.view(1, B * C, L)

        # 对 f_x 进行两侧 Padding，以容纳左右方向的 shift
        f_x_padded = F.pad(f_x_reshaped, (self.max_delay, self.max_delay))

        # 将 f_y 展平为独立的三维权重核，每个核对应一个组
        # F.conv1d 的 weight 维度要求: [out_channels, in_channels/groups, kernel_size]
        # 维度变换: [B, C, L] -> [B*C, 1, L]
        f_y_reshaped = f_y.view(B * C, 1, L)

        # 3. 使用底层 CUDA 算子计算互相关
        # 由于 kernel_size = L，滑动窗口为 2*max_delay + 1
        corr = F.conv1d(f_x_padded, f_y_reshaped, groups=B * C)

        # 4. 恢复输出维度: [1, B*C, 2*max_delay + 1] -> [B, C, 2*max_delay + 1]
        corr = corr.view(B, C, -1)

        return corr


class DC_SWCNet(nn.Module):
    def __init__(self, max_delay=100, feature_dim=32):
        super().__init__()
        self.max_delay = max_delay

        # 将双通道输入分离，但在同一骨干网络中共享权重提取特征
        self.feature_extractor = nn.Sequential(
            NonCausalDilatedConv1d(1, 16, kernel_size=5, dilation=1),
            NonCausalDilatedConv1d(16, 32, kernel_size=5, dilation=2),
            NonCausalDilatedConv1d(32, feature_dim, kernel_size=5, dilation=4),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=8),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=16),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=32),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=64),
        )
        self.channel_attention = ChannelAttention(feature_dim)
        self.correlation_layer = OptimizedFeatureCorrelationLayer(max_delay)

        # 1D 预测头：将多通道相关性压缩为一维最终时延热力图
        self.prediction_head = nn.Sequential(
            nn.Conv1d(feature_dim, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            # nn.Sigmoid()  # 输出 0-1 的概率
        )

    def forward(self, x):
        # x 维度: [B, 2, L]
        x_ch = x[:, 0:1, :]
        y_ch = x[:, 1:2, :]

        # 分别提取高维特征
        f_x = self.feature_extractor(x_ch)
        f_y = self.feature_extractor(y_ch)

        # 利用通道注意力，自动压制发生严重低频串扰的通道，提升干净瞬态通道权重
        f_x = self.channel_attention(f_x)
        f_y = self.channel_attention(f_y)

        # 特征互相关
        corr_matrix = self.correlation_layer(f_x, f_y)  # [B, C, 2*max_delay+1]

        # 输出一维热力图
        heatmap = self.prediction_head(corr_matrix).squeeze(1)  # [B, 2*max_delay+1]
        return heatmap

class WeightedGaussianMSELoss(nn.Module):
    def __init__(self, peak_weight=10.0, bg_weight=1.0):
        super().__init__()
        self.peak_weight = peak_weight
        self.bg_weight = bg_weight

    def forward(self, pred, target):
        # target 是真实的 1D 高斯热力图 (0 到 1 之间)
        # 我们利用 target 本身的值来生成权重图
        # 当 target 接近 1 时，权重为 peak_weight；当 target 为 0 时，权重为 bg_weight
        weight_map = self.bg_weight + target * (self.peak_weight - self.bg_weight)

        # 计算加权的 MSE
        squared_error = (pred - target) ** 2
        weighted_loss = weight_map * squared_error

        return torch.mean(weighted_loss)

class AsymmetricGaussianMSELoss(nn.Module):
    def __init__(self, peak_weight=50.0, bg_weight=1.0, fp_weight=20.0):
        """
        引入伪峰定向打击的非对称 MSE。
        peak_weight: 波峰区域的权重（激励网络寻找目标）
        bg_weight: 正常背景的底噪权重（保持平滑）
        fp_weight: 伪峰惩罚权重（False Positive，定向打击异常突起）
        """
        super().__init__()
        self.peak_weight = peak_weight
        self.bg_weight = bg_weight
        self.fp_weight = fp_weight

    def forward(self, pred, target):
        # 1. 基础平滑权重：目标越接近1，权重越大
        base_weight = self.bg_weight + target * (self.peak_weight - self.bg_weight)

        # 2. 定向定位伪峰 (False Positive Mask)
        # 条件A：真实标签在这里是背景 (比如 target < 0.1)
        is_background = (target < 0.1).float()
        # 条件B：但网络的预测值异常凸起 (比如 pred > 0.2，阈值可根据你的伪峰高度微调)
        is_false_positive = (pred > 0.2).float()

        # 伪峰掩码：同时满足A和B的地方
        fp_mask = is_background * is_false_positive

        # 3. 最终权重融合：在出现伪峰的地方，强行叠加上 fp_weight 的巨额罚款
        final_weight = base_weight + fp_mask * self.fp_weight

        # 4. 计算最终 Loss
        squared_error = (pred - target) ** 2
        weighted_loss = final_weight * squared_error

        return weighted_loss.sum() / pred.size(0)



def extract_peaks_from_heatmap(heatmap, max_delay, threshold=0.5):
    """
    从一维热力图中提取时延峰值（支持多点、亚像素）
    """
    heatmap_np = heatmap.detach().cpu().numpy()
    batch_peaks = []

    for i in range(heatmap_np.shape[0]):
        hm = heatmap_np[i]
        # 寻找局部峰值 (一维 NMS)
        peaks, properties = find_peaks(hm, height=threshold, distance=10)

        sub_pixel_peaks = []
        for p in peaks:
            # 二次插值 (Quadratic Interpolation) 获取亚像素精度
            if 0 < p < len(hm) - 1:
                alpha = hm[p - 1]
                beta = hm[p]
                gamma = hm[p + 1]
                # 抛物线顶点偏移量公式
                offset = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                exact_p = p + offset
            else:
                exact_p = p

            # 还原为真实延迟时间 (-max_delay 到 +max_delay)
            actual_delay = exact_p - max_delay
            sub_pixel_peaks.append(actual_delay)

        batch_peaks.append(sub_pixel_peaks)
        # print(batch_peaks)

    return batch_peaks


def train_and_validate(epochs):
    # 超参数配置
    batch_size = 64
    max_delay = 100
    window_size = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DC_SWCNet(max_delay=max_delay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)

    # 采用高斯平滑后的 MSE Loss
    criterion = WeightedGaussianMSELoss()

    # 划分数据集 (80% 训练集, 20% 测试集)
    train_dataset = DelayEstimationDataset(num_samples=8000, window_size=window_size, max_delay=max_delay)
    test_dataset = DelayEstimationDataset(num_samples=2000, window_size=window_size, max_delay=max_delay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 训练循环 (带进度条)
    train_losses, test_losses = [], []
    print(f"Training on: {device}")

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 测试阶段 ---
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test] ", leave=True)
            for batch_x, batch_y in test_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f">> Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}\n")
                    # ... 前面的训练循环代码 ...

        if (epoch + 1) % 4 == 0:
            # --- 验证阶段 ---
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = generate_simulation_data(4, window_size, max_delay)
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs)

                print("验证批次时延预测结果:")
                predicted_peaks = extract_peaks_from_heatmap(val_outputs, max_delay, threshold=0.4)
                true_peaks = extract_peaks_from_heatmap(val_targets, max_delay, threshold=0.4)

                for b in range(4):
                    print(f"  样本 {b}:")
                    print(f"    真实时延 : {['{:.2f}'.format(p) for p in true_peaks[b]]}")
                    print(f"    预测时延 : {['{:.2f}'.format(p) for p in predicted_peaks[b]]}")

                # ==========================================
                # --- 新增：一维热力图输出可视化 ---
                # ==========================================
                # 将张量转移到 CPU 并转换为 NumPy 数组
                targets_np = val_targets.cpu().numpy()
                outputs_np = val_outputs.cpu().numpy()

                # 构建 X 轴坐标 (从 -max_delay 到 +max_delay)
                x_axis = np.arange(-max_delay, max_delay + 1)

                # 创建 4 个子图，呈垂直排列
                fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.5)  # 增加子图之间的垂直间距

                for i in range(4):
                    # 绘制真实的 Target 高斯曲线 (绿色虚线)
                    axes[i].plot(x_axis, targets_np[i], label='Target (Ground Truth)', color='green',
                                 linestyle='--', linewidth=2)

                    # 绘制网络的 Output 预测曲线 (红色实线)
                    axes[i].plot(x_axis, outputs_np[i], label='Prediction (Output)', color='red', alpha=0.8,
                                 linewidth=2)

                    # 用垂直点划线标出提取到的确切峰值位置
                    for tp in true_peaks[i]:
                        axes[i].axvline(x=tp, color='green', linestyle=':', alpha=0.6)
                    for pp in predicted_peaks[i]:
                        axes[i].axvline(x=pp, color='red', linestyle=':', alpha=0.6)

                    # 设置图表格式
                    axes[i].set_title(f"Sample {i} - Heatmap Comparison")
                    axes[i].set_xlabel("Time Delay (Sampling Points)")
                    axes[i].set_ylabel("Probability Intensity")
                    axes[i].legend(loc="upper right")
                    axes[i].grid(True, alpha=0.3)

                    # 固定 Y 轴范围，防止由于预测值过小导致坐标轴自动缩放引发的视觉误判
                    axes[i].set_ylim(-0.1, 1.2)

                plt.show()

                # 释放内存，防止在长时间训练中因为重复创建绘图对象导致 OOM
                plt.close(fig)

                print("-" * 40)

    # 打包需要保存的所有关键组件
    checkpoint = {
        'epoch': epoch,                        # 当前轮次
        'model_state_dict': model.state_dict(), # 模型权重
        'optimizer_state_dict': optimizer.state_dict(), # 优化器状态（动量等）
        'loss': loss,                          # 当前损失值
    }

    # 保存断点文件
    BASE_DIR = './model'
    os.makedirs(BASE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, f'checkpoint_epoch_{epoch}.pth')


    torch.save(checkpoint, MODEL_SAVE_PATH)
    return MODEL_SAVE_PATH


def continue_training(MODEL_SAVE_PATH, total_epochs):
    # 5. 继续训练循环
    batch_size = 64
    max_delay = 100
    window_size = 4000
    epochs = total_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DC_SWCNet(max_delay=max_delay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 采用高斯平滑后的 MSE Loss
    criterion = AsymmetricGaussianMSELoss()

    # 划分数据集 (80% 训练集, 20% 测试集)
    train_dataset = DelayEstimationDataset(num_samples=8000, window_size=window_size, max_delay=max_delay)
    test_dataset = DelayEstimationDataset(num_samples=2000, window_size=window_size, max_delay=max_delay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 训练循环 (带进度条)
    train_losses, test_losses = [], []
    print(f"Training on: {device}")

    # 1. 加载断点文件（使用 map_location 确保在不同设备间兼容）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)

    # 2. 按顺序恢复状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # 4. 获取断点时的轮次，从下一个轮次继续
    start_epoch = checkpoint['epoch'] + 1
    print(f"从第 {start_epoch} 轮继续训练...")


    for epoch in range(start_epoch, total_epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 测试阶段 ---
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test] ", leave=True)
            for batch_x, batch_y in test_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f">> Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}\n")
                    # ... 前面的训练循环代码 ...

        if (epoch + 1) % 4 == 0:
            # --- 验证阶段 ---
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = generate_simulation_data(4, window_size, max_delay)
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs)

                print("验证批次时延预测结果:")
                predicted_peaks = extract_peaks_from_heatmap(val_outputs, max_delay, threshold=0.4)
                true_peaks = extract_peaks_from_heatmap(val_targets, max_delay, threshold=0.4)

                for b in range(4):
                    print(f"  样本 {b}:")
                    print(f"    真实时延 : {['{:.2f}'.format(p) for p in true_peaks[b]]}")
                    print(f"    预测时延 : {['{:.2f}'.format(p) for p in predicted_peaks[b]]}")

                # ==========================================
                # --- 新增：一维热力图输出可视化 ---
                # ==========================================
                # 将张量转移到 CPU 并转换为 NumPy 数组
                targets_np = val_targets.cpu().numpy()
                outputs_np = val_outputs.cpu().numpy()

                # 构建 X 轴坐标 (从 -max_delay 到 +max_delay)
                x_axis = np.arange(-max_delay, max_delay + 1)

                # 创建 4 个子图，呈垂直排列
                fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.5)  # 增加子图之间的垂直间距

                for i in range(4):
                    # 绘制真实的 Target 高斯曲线 (绿色虚线)
                    axes[i].plot(x_axis, targets_np[i], label='Target (Ground Truth)', color='green',
                                 linestyle='--', linewidth=2)

                    # 绘制网络的 Output 预测曲线 (红色实线)
                    axes[i].plot(x_axis, outputs_np[i], label='Prediction (Output)', color='red', alpha=0.8,
                                 linewidth=2)

                    # 用垂直点划线标出提取到的确切峰值位置
                    for tp in true_peaks[i]:
                        axes[i].axvline(x=tp, color='green', linestyle=':', alpha=0.6)
                    for pp in predicted_peaks[i]:
                        axes[i].axvline(x=pp, color='red', linestyle=':', alpha=0.6)

                    # 设置图表格式
                    axes[i].set_title(f"Sample {i} - Heatmap Comparison")
                    axes[i].set_xlabel("Time Delay (Sampling Points)")
                    axes[i].set_ylabel("Probability Intensity")
                    axes[i].legend(loc="upper right")
                    axes[i].grid(True, alpha=0.3)

                    # 固定 Y 轴范围，防止由于预测值过小导致坐标轴自动缩放引发的视觉误判
                    axes[i].set_ylim(-0.1, 1.2)

                plt.show()

                # 释放内存，防止在长时间训练中因为重复创建绘图对象导致 OOM
                plt.close(fig)

                print("-" * 40)

    # 打包需要保存的所有关键组件
    checkpoint = {
        'epoch': epoch,                        # 当前轮次
        'model_state_dict': model.state_dict(), # 模型权重
        'optimizer_state_dict': optimizer.state_dict(), # 优化器状态（动量等）
        'loss': loss,                          # 当前损失值
    }

    # 保存断点文件
    BASE_DIR = './model'
    os.makedirs(BASE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, f'checkpoint_epoch_{epoch}.pth')


    torch.save(checkpoint, MODEL_SAVE_PATH)
    return MODEL_SAVE_PATH


def check_network_is_alive(model):
    """
    检查网络是否有梯度流动的探针
    """
    has_dead_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            if grad_mean < 1e-7:
                print(f"⚠️ 警告: {name} 层梯度极小 ({grad_mean:.2e})，可能已死锁。")
                has_dead_grad = True
    if not has_dead_grad:
        print("✅ 梯度流动正常。")

# --- 单样本强制过拟合测试 ---
def overfit_single_batch_test(model):

    batch_size = 32
    max_delay = 100
    window_size = 5000
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DC_SWCNet(max_delay=max_delay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 1. 生成并固定一个 Batch 的数据（千万不要放在循环里重新生成）
    fixed_inputs, fixed_targets = generate_simulation_data(batch_size=4, window_size=4000, max_delay=100)
    fixed_inputs = fixed_inputs.to(device)
    fixed_targets = fixed_targets.to(device)

    print("开始强制过拟合单 Batch 测试...")
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(fixed_inputs)

        # 2. 如果去掉了 Sigmoid，预测值可能为负，可以用 ReLU 或绝对值保护一下
        # outputs = torch.relu(outputs)

        loss = criterion(outputs, fixed_targets)
        loss.backward()

        # 3. 观察梯度探针（每 50 轮看一次）
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch+1}, Loss: {loss.item():.6f}")
            check_network_is_alive(model)

        optimizer.step()
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_simulation_data(batch_size, window_size=4000, max_delay=100, max_points=3):
    """
    生成模拟的 MZ-Sagnac 多点重叠扰动数据及一维高斯热力图标签
    采样率参考 5 MSa/s [cite: 825]。
    """
    X = torch.zeros(batch_size, 2, window_size)
    # 标签热力图：长度为 2 * max_delay + 1 的一维数组
    Y_heatmap = torch.zeros(batch_size, 2 * max_delay + 1)

    t = np.arange(window_size)

    for b in range(batch_size):
        num_points = np.random.randint(1, max_points + 1)
        x_signal = np.zeros(window_size)
        y_signal = np.zeros(window_size)

        for p in range(num_points):
            # 模拟随机延迟和随机起振时间，制造严重的 Temporal Overlap
            delay = np.random.randint(-max_delay + 10, max_delay - 10)
            onset = np.random.randint(100, window_size - 1000)

            # 多频混合瞬态衰减信号
            A = np.random.uniform(0.5, 1.5)
            alpha = np.random.uniform(0.001, 0.005)
            freqs = [np.random.uniform(0.01, 0.05), np.random.uniform(0.1, 0.2)]

            # 构造 x(t)
            x_component = np.zeros(window_size)
            envelope = A * np.exp(-alpha * (t[onset:] - onset))
            for f in freqs:
                x_component[onset:] += envelope * np.sin(2 * np.pi * f * (t[onset:] - onset))

            # 构造 y(t) = x(t - delay)
            y_component = np.roll(x_component, delay)

            x_signal += x_component
            y_signal += y_component

            # 在对应延迟位置渲染一维高斯分布 (Sigma = 2.0)
            # 避免直接回归造成的偏移，利用高斯分布引导网络学习平滑特征
            center = delay + max_delay
            sigma = 2.0
            x_grid = np.arange(2 * max_delay + 1)
            gaussian = np.exp(-((x_grid - center) ** 2) / (2 * sigma ** 2))

            # 取多点高斯峰的最大值（如果峰重叠）
            # 这里认为当多个峰同时存在时，选择最大者，
            # 网络最终的输出要和这个热力图进行拟合
            Y_heatmap[b] = torch.max(Y_heatmap[b], torch.tensor(gaussian, dtype=torch.float32))

        # 添加高斯白噪声
        X[b, 0, :] = torch.tensor(x_signal) + torch.randn(window_size) * 0.05
        X[b, 1, :] = torch.tensor(y_signal) + torch.randn(window_size) * 0.05

    return X, Y_heatmap


class NonCausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # 严格计算对称 Padding，确保非因果性（零相位偏移）
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(1,out_channels,1e-8)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FeatureCorrelationLayer(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

    def forward(self, f_x, f_y):
        """
        计算特征级别的互相关。
        输入维度: [B, C, L]
        输出维度: [B, C, 2*max_delay + 1]
        """
        B, C, L = f_x.size()
        correlations = []

        # 归一化特征，类似于原文公式(7)
        f_x = F.normalize(f_x, p=2, dim=2)
        f_y = F.normalize(f_y, p=2, dim=2)

        # 在特征维度进行不同 shift 的滑动点乘
        for shift in range(-self.max_delay, self.max_delay + 1):
            if shift < 0:
                x_shift = f_x[:, :, :shift]
                y_shift = f_y[:, :, -shift:]
            elif shift > 0:
                x_shift = f_x[:, :, shift:]
                y_shift = f_y[:, :, :-shift]
            else:
                x_shift = f_x
                y_shift = f_y

            # 计算相似度并对时间维度求和
            corr = torch.sum(x_shift * y_shift, dim=2, keepdim=True)  # [B, C, 1]
            correlations.append(corr)

        return torch.cat(correlations, dim=2)  # [B, C, 2*max_delay + 1]


class OptimizedFeatureCorrelationLayer(nn.Module):
    def __init__(self, max_delay):
        super().__init__()
        self.max_delay = max_delay

    def forward(self, f_x, f_y):
        """
        利用 F.conv1d 和 Grouped Convolution 极致优化计算图的互相关
        输入维度: f_x, f_y -> [B, C, L]
        输出维度: [B, C, 2*max_delay + 1]
        """
        B, C, L = f_x.size()

        # 1. 归一化 (计算 Cosine 相似度必备)
        f_x = F.normalize(f_x, p=2, dim=2)
        f_y = F.normalize(f_y, p=2, dim=2)

        # 2. 维度重构技巧
        # 互相关是在 x 上滑动 y。为了用 F.conv1d 批量处理，
        # 我们把 f_x 作为 "Input"，把 f_y 作为 "Weight"。
        # 为了不让不同 Batch 和不同 Channel 相互干扰，我们使用 分组卷积 (groups = B * C)

        # 将 f_x 展平为单 Batch，包含 B*C 个独立的组
        # 维度变换: [B, C, L] -> [1, B*C, L]
        f_x_reshaped = f_x.view(1, B * C, L)

        # 对 f_x 进行两侧 Padding，以容纳左右方向的 shift
        f_x_padded = F.pad(f_x_reshaped, (self.max_delay, self.max_delay))

        # 将 f_y 展平为独立的三维权重核，每个核对应一个组
        # F.conv1d 的 weight 维度要求: [out_channels, in_channels/groups, kernel_size]
        # 维度变换: [B, C, L] -> [B*C, 1, L]
        f_y_reshaped = f_y.view(B * C, 1, L)

        # 3. 使用底层 CUDA 算子计算互相关
        # 由于 kernel_size = L，滑动窗口为 2*max_delay + 1
        corr = F.conv1d(f_x_padded, f_y_reshaped, groups=B * C)

        # 4. 恢复输出维度: [1, B*C, 2*max_delay + 1] -> [B, C, 2*max_delay + 1]
        corr = corr.view(B, C, -1)

        return corr


class DC_SWCNet(nn.Module):
    def __init__(self, max_delay=100, feature_dim=32):
        super().__init__()
        self.max_delay = max_delay

        # 将双通道输入分离，但在同一骨干网络中共享权重提取特征
        self.feature_extractor = nn.Sequential(
            NonCausalDilatedConv1d(1, 16, kernel_size=5, dilation=1),
            NonCausalDilatedConv1d(16, 32, kernel_size=5, dilation=2),
            NonCausalDilatedConv1d(32, feature_dim, kernel_size=5, dilation=4),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=8),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=16),
            NonCausalDilatedConv1d(feature_dim, feature_dim, kernel_size=5, dilation=32),
        )

        self.channel_attention = ChannelAttention(feature_dim)
        self.correlation_layer = FeatureCorrelationLayer(max_delay)

        # 1D 预测头：将多通道相关性压缩为一维最终时延热力图
        self.prediction_head = nn.Sequential(
            nn.Conv1d(feature_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 0-1 的概率
        )

    def forward(self, x):
        # x 维度: [B, 2, L]
        x_ch = x[:, 0:1, :]
        y_ch = x[:, 1:2, :]

        # 分别提取高维特征
        f_x = self.feature_extractor(x_ch)
        f_y = self.feature_extractor(y_ch)

        # 利用通道注意力，自动压制发生严重低频串扰的通道，提升干净瞬态通道权重
        f_x = self.channel_attention(f_x)
        f_y = self.channel_attention(f_y)

        # 特征互相关
        corr_matrix = self.correlation_layer(f_x, f_y)  # [B, C, 2*max_delay+1]

        # 输出一维热力图
        heatmap = self.prediction_head(corr_matrix).squeeze(1)  # [B, 2*max_delay+1]
        return heatmap

def extract_peaks_from_heatmap(heatmap, max_delay, threshold=0.5):
    """
    从一维热力图中提取时延峰值（支持多点、亚像素）
    """
    heatmap_np = heatmap.detach().cpu().numpy()
    batch_peaks = []

    for i in range(heatmap_np.shape[0]):
        hm = heatmap_np[i]
        # 寻找局部峰值 (一维 NMS)
        peaks, properties = find_peaks(hm, height=threshold, distance=10)

        sub_pixel_peaks = []
        for p in peaks:
            # 二次插值 (Quadratic Interpolation) 获取亚像素精度
            if 0 < p < len(hm) - 1:
                alpha = hm[p - 1]
                beta = hm[p]
                gamma = hm[p + 1]
                # 抛物线顶点偏移量公式
                offset = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                exact_p = p + offset
            else:
                exact_p = p

            # 还原为真实延迟时间 (-max_delay 到 +max_delay)
            actual_delay = exact_p - max_delay
            sub_pixel_peaks.append(actual_delay)

        batch_peaks.append(sub_pixel_peaks)
        # print(batch_peaks)

    return batch_peaks


def train_and_validate():
    # 超参数配置
    batch_size = 32
    max_delay = 100
    window_size = 4000
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DC_SWCNet(max_delay=max_delay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 采用高斯平滑后的 MSE Loss
    criterion = nn.MSELoss()

    print("开始训练 DC-ASWC-Net...")
    for epoch in tqdm(range(epochs)):
        model.train()
        # 在线生成增强的仿真数据
        inputs, targets = generate_simulation_data(batch_size, window_size, max_delay)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            # ... 前面的训练循环代码 ...

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.6f}")

            # --- 验证阶段 ---
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = generate_simulation_data(4, window_size, max_delay)
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs)

                print("验证批次时延预测结果:")
                predicted_peaks = extract_peaks_from_heatmap(val_outputs, max_delay, threshold=0.4)
                true_peaks = extract_peaks_from_heatmap(val_targets, max_delay, threshold=0.4)

                for b in range(4):
                    print(f"  样本 {b}:")
                    print(f"    真实时延 : {['{:.2f}'.format(p) for p in true_peaks[b]]}")
                    print(f"    预测时延 : {['{:.2f}'.format(p) for p in predicted_peaks[b]]}")

                # ==========================================
                # --- 新增：一维热力图输出可视化 ---
                # ==========================================
                # 将张量转移到 CPU 并转换为 NumPy 数组
                targets_np = val_targets.cpu().numpy()
                outputs_np = val_outputs.cpu().numpy()

                # 构建 X 轴坐标 (从 -max_delay 到 +max_delay)
                x_axis = np.arange(-max_delay, max_delay + 1)

                # 创建 4 个子图，呈垂直排列
                fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.5)  # 增加子图之间的垂直间距

                for i in range(4):
                    # 绘制真实的 Target 高斯曲线 (绿色虚线)
                    axes[i].plot(x_axis, targets_np[i], label='Target (Ground Truth)', color='green',
                                 linestyle='--', linewidth=2)

                    # 绘制网络的 Output 预测曲线 (红色实线)
                    axes[i].plot(x_axis, outputs_np[i], label='Prediction (Output)', color='red', alpha=0.8,
                                 linewidth=2)

                    # 用垂直点划线标出提取到的确切峰值位置
                    for tp in true_peaks[i]:
                        axes[i].axvline(x=tp, color='green', linestyle=':', alpha=0.6)
                    for pp in predicted_peaks[i]:
                        axes[i].axvline(x=pp, color='red', linestyle=':', alpha=0.6)

                    # 设置图表格式
                    axes[i].set_title(f"Sample {i} - Heatmap Comparison")
                    axes[i].set_xlabel("Time Delay (Sampling Points)")
                    axes[i].set_ylabel("Probability Intensity")
                    axes[i].legend(loc="upper right")
                    axes[i].grid(True, alpha=0.3)

                    # 固定 Y 轴范围，防止由于预测值过小导致坐标轴自动缩放引发的视觉误判
                    axes[i].set_ylim(-0.1, 1.2)

                plt.show()

                # 释放内存，防止在长时间训练中因为重复创建绘图对象导致 OOM
                plt.close(fig)

                print("-" * 40)
                    
                    
                    
                    
            print("-" * 40)


if __name__ == "__main__":
    train_and_validate()
>>>>>>> c88a307 (cloud codes)
