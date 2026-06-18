import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNResidualBlock(nn.Module):
    """
    非因果 TCN 残差块
    包含: 空洞卷积 -> 归一化 -> 激活 -> Dropout -> (重复) + 残差连接
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()

        # 非因果填充 (Same Padding)，保证输入输出时间步长度一致
        # 要求 kernel_size 必须为奇数
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.GELU()  # 相比 ReLU，GELU 在信号处理特征提取中表现通常更平滑
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.norm1, self.relu1, self.drop1,
            self.conv2, self.norm2, self.relu2, self.drop2
        )

        # 匹配残差连接的通道数
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class DenseTCN(nn.Module):
    """
    用于时延估计的非因果 TCN 网络
    输入: (Batch, 2, Seq_Len) -> 通道1: 原始信号, 通道2: 延迟信号
    输出: (Batch, Seq_Len) 的时延估计值与对应概率
    """

    def __init__(self, num_inputs=2, num_channels=[64, 64, 128, 128],
                 kernel_size=3, dropout=0.2, max_delay=50):
        super(DenseTCN, self).__init__()

        assert kernel_size % 2 == 1, "Kernel size 必须为奇数以实现对称的 Same Padding"

        self.max_delay = max_delay
        # 类别数为从 -max_delay 到 +max_delay，共 2 * max_delay + 1 个可能值
        self.num_classes = 2 * max_delay + 1

        self.tcn_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()  # 用于统一跳跃连接的通道数

        # 构建堆叠的 TCN 块，膨胀系数呈指数增长
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            self.tcn_blocks.append(
                TCNResidualBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)
            )
            # 1x1 卷积用于将每个 block 的输出映射到相同的维度以便求和 (跳跃连接)
            self.skip_convs.append(nn.Conv1d(out_channels, num_channels[-1], 1))

        # 最终分类头
        self.final_conv = nn.Conv1d(num_channels[-1], self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        x shape: (Batch_size, 2, Sequence_Length)
        """
        x = x.permute(0, 2, 1)
        skip_connections = 0

        # 前向传播并收集跳跃连接
        for tcn_block, skip_conv in zip(self.tcn_blocks, self.skip_convs):
            x = tcn_block(x)
            skip_connections = skip_connections + skip_conv(x)

        # (Batch_size, num_classes, Sequence_Length)
        logits = self.final_conv(skip_connections)

        # 在类别维度(dim=1)上计算 Softmax 得到概率
        probs_distribution = F.softmax(logits, dim=1)

        # 提取最高概率及其对应的索引 (相当于滑动窗口的估计)
        # max_probs shape: (Batch_size, Sequence_Length)
        # est_indices shape: (Batch_size, Sequence_Length)
        max_probs, est_indices = torch.max(probs_distribution, dim=1)

        # 将索引 (0 到 2*max_delay) 转换为实际的时延值 (-max_delay 到 +max_delay)
        estimated_delays = est_indices - self.max_delay

        return estimated_delays, max_probs, probs_distribution

class TDEJointLoss(nn.Module):
    """
    时延估计联合损失函数
    结合了 Cross-Entropy (提供尖锐的峰值分类能力) 和
    Soft-Argmax MSE (提供类别间的物理距离惩罚，引导梯度向正确方向移动)
    """

    def __init__(self, max_delay, alpha=1.0, beta=0.1, label_smoothing=0.05):
        super(TDEJointLoss, self).__init__()
        self.max_delay = max_delay
        self.alpha = alpha  # 交叉熵损失的权重
        self.beta = beta  # MSE 回归损失的权重

        # PyTorch 的 CE Loss 原生支持 label_smoothing (需版本 >= 1.10)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse_loss = nn.MSELoss()

        # 生成类别对应的物理时延值向量: [-max_delay, ..., 0, ..., max_delay]
        # shape: (num_classes,)
        delay_values = torch.arange(-max_delay, max_delay + 1, dtype=torch.float32)
        self.register_buffer('delay_values', delay_values)

    def forward(self, logits, target_delays):
        """
        参数:
        - logits: 模型的原始输出, shape (Batch, Num_Classes, Seq_Length)
        - target_delays: 真实的时延值, shape (Batch, Seq_Length), 取值范围在 [-max_delay, max_delay]
        """
        # --- 1. 准备数据 ---
        # target_delays 是真实的物理时延，我们需要将其转换为分类的 index (0 到 2*max_delay)
        target_indices = target_delays + self.max_delay
        target_indices = target_indices.long()

        # --- 2. 计算分类损失 (Cross Entropy) ---
        # logits shape: (B, C, L), target_indices shape: (B, L)
        # PyTorch 的 CrossEntropyLoss 天然支持多维输入，C 必须在第二维
        loss_ce = self.ce_loss(logits, target_indices)

        # --- 3. 计算回归损失 (Soft-Argmax MSE) ---
        # 计算每个类别的概率分布 shape: (B, C, L)
        probs = F.softmax(logits, dim=1)

        # 为了进行点乘，将 delay_values 扩展维度: (C,) -> (1, C, 1)
        v = self.delay_values.view(1, -1, 1)

        # 计算软期望 (在类别维度 dim=1 上求和)
        # expected_delays shape: (B, L)
        expected_delays = torch.sum(probs * v, dim=1)

        # 计算与真实物理时延的均方误差
        loss_mse = self.mse_loss(expected_delays, target_delays.float())

        # --- 4. 联合总损失 ---
        total_loss = self.alpha * loss_ce + self.beta * loss_mse

        return total_loss, loss_ce, loss_mse
