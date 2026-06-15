import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 暂时用不到
class CausalConv1d(nn.Module):
    """非因果1D卷积（实际为对称因果卷积，用于构建非因果TCN）"""
    # TODO : 补零方式选取
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,pad_mode=1):
        super(CausalConv1d, self).__init__()
        # 计算对称padding，确保感受野对称
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode= "zeros" if pad_mode else "circular",
        )

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        x = self.conv(x)
        # 确保输出长度与输入相同（对称卷积）
        return x


class GLN(nn.Module):
    """
    实现非因果模型全局层归一化
    """
    # TODO ：最新研究使用RMS层归一化
    def __init__(self, in_channels, groups=1):
        super(GLN, self).__init__()
        self.gln = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-8, affine=True)

    def forward(self, x):
        return self.gln(x)

class ResidualConvBlock(nn.Module):
    """卷积残差块，包含深度可分离膨胀卷积、非线性激活，层归一化和残差，跳跃连接"""

    def __init__(self, input_channel, hidden_channel,padding_size, dilation, kernel_size=3, skip=True):
        super(ResidualConvBlock, self).__init__()

        self.dilation = dilation
        self.skip = skip

        self.bottle_neck = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding_size
        self.depth_wise_conv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel_size, dilation=dilation,
                                           groups=hidden_channel, padding=self.padding)
        # TODO swish激活函数
        self.act = nn.PReLU()
        self.norm = GLN(hidden_channel, groups=1)

        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        """
        依次为：1x1 bottlneck ->act-> gln -> D-conv1d->act ->gln -> 1x1 conv
                                                        skip_connection
        # TODO : 最后输出没有激活层
        """
        output = self.norm(self.act(self.bottle_neck(input)))
        output = self.norm(self.act(self.depth_wise_conv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual



# class OldTCN(nn.Module):
#     def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
#                  layer, stack, kernel=3, skip=True):
#         super(OldTCN, self).__init__()
#
#         # input is a sequence of features of shape (B, N, L)
#         self.skip = skip
#         # normalization
#
#         self.norm = GLN(input_dim, 1)
#
#         self.bottle_neck = nn.Conv1d(input_dim, BN_dim, 1)
#
#         # TCN for feature extraction
#         self.receptive_field = 0
#
#         self.TCN = nn.ModuleList([])
#         for s in range(stack):
#             for i in range(layer):
#                 if self.dilated:
#                     self.TCN.append(ResidualConvBlock(BN_dim, hidden_dim, padding_size=2 ** i,dilation=2 ** i,kernel_size= kernel, skip=skip,))
#                 if i == 0 and s == 0:
#                     self.receptive_field += kernel
#                 else:
#                     if self.dilated:
#                         self.receptive_field += (kernel - 1) * 2 ** i
#
#         # print("Receptive field: {:3d} frames.".format(self.receptive_field))
#
#         # output layer
#
#         self.output = nn.Sequential(nn.PReLU(),
#                                     nn.Conv1d(BN_dim, output_dim, 1))
#
#
#
#     def forward(self, input):
#
#         # input shape: (B, N, L)
#
#         # normalization
#         output = self.BN(self.LN(input))
#
#         # pass to TCN
#         if self.skip:
#             skip_connection = 0.
#             for i in range(len(self.TCN)):
#                 residual, skip = self.TCN[i](output)
#                 output = output + residual
#                 skip_connection = skip_connection + skip
#         else:
#             for i in range(len(self.TCN)):
#                 residual = self.TCN[i](output)
#                 output = output + residual
#
#         # output layer
#         if self.skip:
#             output = self.output(skip_connection)
#         else:
#             output = self.output(output)
#
#         return output

class TCN(nn.Module):
    """一维非因果TCN时延估计网络"""
    def __init__(self,input_dim,enc_dim, bottle_neck_dim, hidden_dim,layer, stack, kernel=3,
                 skip=True,sources=1, max_delay=50):
        """
        参数:
        - input_dim: 输入通道数（默认2：原始信号+延迟信号）
        output_dim = enc_dim * num_spk
        bottle_neck_dim = feature_dim
        hidden_dim = feature_dim * 4
        - hidden_dim: 隐藏层通道数
        - kernel_size: 卷积核大小
        - max_delay: 最大可能时延（决定输出维度）
        """
        super(TCN, self).__init__()
        self.skip = skip
        self.max_delay = max_delay
        self.output_size = 2 * max_delay + 1  # 时延范围: [-max_delay, max_delay]

        # 初始投影层（将2通道映射到隐藏维度）
        self.start_conv = nn.Conv1d(input_dim, enc_dim, kernel_size=1, bias=False)
        # TODO : 是否需要滑动窗口
        # self.start_conv = nn.Conv1d(input_dim,enc_dim,kernel_size=20,stride=10,padding_mode='zeros',bias=False)


        self.norm = GLN(enc_dim, 1)
        self.act = nn.PReLU()
        self.bottle_neck = nn.Conv1d(enc_dim, bottle_neck_dim, 1)
        self.de_bottle_neck = nn.Conv1d(bottle_neck_dim, hidden_dim, 1)
        # TCN for feature extraction
        self.receptive_field = 0

        self.TCNStack = nn.ModuleList([])

        for s in range(stack):
            for i in range(layer):
                dil = 2 ** i
                padding = (kernel - 1) * dil // 2
                self.TCNStack.append(
                ResidualConvBlock(bottle_neck_dim, hidden_dim, padding_size=padding, dilation=dil, kernel_size=kernel,
                                          skip=skip, ))
                if i == 0 and s == 0:
                    self.receptive_field += kernel

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # 输出层（映射到时延分布）
        self.end_conv = nn.Conv1d(hidden_dim, self.output_size, 1)

        # 激活函数（输出为概率分布）
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        输入x形状: (batch_size, sequence_length, 2)
        输出形状: (batch_size, output_size) - 时延概率分布
        """
        # 调整维度顺序: (batch, channels, seq_len)
        input = input.permute(0, 2, 1)
        # TODO 调整序列长度，满足卷积条件
        # 初始投影
        input = self.start_conv(input) # (batch, enc_dim, seq_len)

        # normalization
        output = self.bottle_neck(self.norm(input))

        # pass to TCN
        skip_connection = 0.
        if self.skip:
            for i in range(len(self.TCNStack)):
                residual, skip = self.TCNStack[i](output)
                output = output + residual
                skip_connection = skip_connection + skip #(batch, hidden_dim, seq_len)
        else:
            for i in range(len(self.TCNStack)):
                residual = self.TCNStack[i](output)
                output = output + residual

        # 输出层
        if self.skip:
            skip_connection = self.de_bottle_neck(self.act(skip_connection))
            skip_connection = skip_connection.mean(dim=-1, keepdim=True) # (batch, hidden_dim, 1)
            output = self.end_conv(skip_connection) # (batch, delays, 1)
        else:
            output = self.de_bottle_neck(self.act(output))
            output = output.mean(dim=-1, keepdim=True)
            output = self.end_conv(output)

        output = output.squeeze(-1)  # (batch, output_size)

        # 归一化为概率分布
        return self.softmax(output)

        # 通过TCN块
        # skip_connections = []
        # for block in self.tcn_blocks:
        #     x = block(x)
        #     # 收集跳跃连接特征
        #     skip_connections.append(self.global_pool(x).squeeze(-1))
        #
        # # 聚合跳跃连接
        # if skip_connections:
        #     x = torch.stack(skip_connections, dim=-1).mean(dim=-1)
        # else:
        #     x = self.global_pool(x).squeeze(-1)





def create_gaussian_label(delay, max_delay, sigma=2.0):
    """
    创建高斯软标签
    delay: 真实时延（标量或张量）
    max_delay: 最大可能时延
    sigma: 高斯分布标准差
    """
    device = delay.device if isinstance(delay, torch.Tensor) else torch.device('cpu')

    # 创建时延轴: [-max_delay, ..., 0, ..., max_delay]
    delays = torch.arange(-max_delay, max_delay + 1, device=device, dtype=torch.float32)

    # 计算高斯分布 (batch_size, output_size)
    if len(delay.shape) == 0:  # 单个延迟值
        delay = delay.unsqueeze(0)

    # 扩展维度以便广播
    delay = delay.view(-1, 1)

    # 高斯分布
    gaussian = torch.exp(-0.5 * ((delays - delay) / sigma) ** 2)
    # 归一化为概率分布
    gaussian = gaussian / gaussian.sum(dim=1, keepdim=True)

    return gaussian


def tcn_loss(prediction, delay, max_delay, sigma=2.0, lambda_gauss=0.7, lambda_l1=0.3):
    """
    复合损失函数
    prediction: 网络预测的时延分布 (batch_size, output_size)
    delay: 真实时延 (batch_size,)
    max_delay: 最大可能时延
    sigma: 高斯分布标准差
    lambda_gauss: 高斯分布误差权重
    lambda_l1: 期望L1误差权重
    """
    # 1. 高斯分布误差 (MSE between predicted and target distributions)
    target_gauss = create_gaussian_label(delay, max_delay, sigma)
    gauss_loss = F.mse_loss(prediction, target_gauss)

    # 2. 期望L1误差
    # 创建时延轴
    delays = torch.arange(-max_delay, max_delay + 1, device=prediction.device, dtype=torch.float32)
    # 计算预测分布的期望时延
    expected_delay = torch.sum(prediction * delays, dim=1)
    # 计算L1误差
    l1_loss = F.smooth_l1_loss(expected_delay, delay.float())

    # 组合损失
    total_loss = lambda_gauss * gauss_loss + lambda_l1 * l1_loss

    # return total_loss, {
    #     'gauss_loss': gauss_loss.item(),
    #     'l1_loss': l1_loss.item(),
    #     'expected_delay': expected_delay.detach()
    # }
    return total_loss


# 示例用法
if __name__ == "__main__":
    # 参数设置
    MAX_DELAY = 50
    SEQ_LENGTH = 1000
    BATCH_SIZE = 32

    # 创建网络
    model = TCN(
        input_dim=2,enc_dim=64,bottle_neck_dim=16,hidden_dim=64,layer=6,stack=2,max_delay=MAX_DELAY,
    )


    # 模拟数据
    def generate_sample_sequence(seq_length, delay):
        """生成带有时延的信号对"""
        # 生成随机噪声作为原始信号
        signal = np.random.randn(seq_length)

        # 创建延迟版本（简单循环移位）
        delayed_signal = np.roll(signal, delay)

        # 组合成双通道输入
        input_data = np.stack([signal, delayed_signal], axis=-1)

        return input_data, delay


    # 生成一批训练数据
    def generate_batch(batch_size, seq_length, max_delay):
        inputs = []
        delays = []

        for _ in range(batch_size):
            # 随机生成时延（在范围内）
            delay = np.random.randint(-max_delay, max_delay + 1)
            input_data, _ = generate_sample_sequence(seq_length, delay)
            inputs.append(input_data)
            delays.append(delay)

        return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(delays), dtype=torch.float32)


    # 测试前向传播
    inputs, delays = generate_batch(BATCH_SIZE, SEQ_LENGTH, MAX_DELAY)
    outputs = model(inputs)

    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {outputs.shape} (对应时延范围: {-MAX_DELAY} 到 {MAX_DELAY})")

    # 测试损失函数
    loss, loss_details = tcn_loss(outputs, delays, MAX_DELAY)
    print(f"\n损失计算:")
    print(f"总损失: {loss.item():.6f}")
    print(f"高斯分布误差: {loss_details['gauss_loss']:.6f}")
    print(f"L1误差: {loss_details['l1_loss']:.6f}")

    # 可视化示例
    plt.figure(figsize=(12.0,8.0))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 2. 解决负号 '-' 显示为方块的问题（强烈建议加上）
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 可视化输入信号
    plt.subplot(3, 1, 1)
    plt.plot(inputs[0, :, 0].numpy(), 'b', label='原始信号')
    plt.plot(inputs[0, :, 1].numpy(), 'r--', label='延迟信号')
    plt.title(f'输入信号 (真实时延: {delays[0].item()})')
    plt.legend()

    # 2. 可视化目标高斯分布
    plt.subplot(3, 1, 2)
    target_gauss = create_gaussian_label(delays[0:1], MAX_DELAY, sigma=2.0)
    delays_axis = np.arange(-MAX_DELAY, MAX_DELAY + 1)
    plt.plot(delays_axis, target_gauss[0].detach().numpy(), 'g-', linewidth=2)
    plt.axvline(x=delays[0].item(), color='r', linestyle='--', alpha=0.7)
    plt.title('目标高斯分布 (真实时延标记为红线)')
    plt.grid(True)

    # 3. 可视化预测分布
    plt.subplot(3, 1, 3)
    plt.plot(delays_axis, outputs[0].detach().numpy(), 'b-', linewidth=2)
    plt.axvline(x=loss_details['expected_delay'][0].item(), color='r', linestyle='--', alpha=0.7)
    plt.title(f'预测分布 (预测时延: {loss_details["expected_delay"][0].item():.2f})')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('tcn_delay_estimation.png')
    print("\n已生成可视化结果: tcn_delay_estimation.png")