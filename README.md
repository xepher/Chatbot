# 聊天机器人

聊天机器人的 PyTorch 实现。


## 依赖与环境管理

本项目使用 [uv](https://github.com/astral-sh/uv) 进行 Python 依赖与环境管理。

主要依赖版本：
- Python >= 3.11
- PyTorch >= 2.12
- [jieba 0.42.1](https://github.com/fxsjy/jieba)
- [tqdm 4.68.4](https://github.com/tqdm/tqdm)

安装项目依赖及初始化虚拟环境：
```bash
$ uv sync
```

## 🚀 性能调优与去过拟合优化

本项目经过系统级的性能调优和算法调优，实现了训练高吞吐与验证高保真的双向提升：

1. **训练提速（5x+ 加速）**：
   * **异步 DataLoader**：引入多进程数据加载，采用 `pin_memory=True` 锁页内存，让数据准备与 GPU 运算完全并行。
   * **工作进程持久化**：启用 `persistent_workers` 参数，规避了 Epoch 切换时后台进程频繁销毁与重构的 4 秒时延。
   * **自动混合精度 (AMP)**：采用统一的新版 `torch.amp` (FP16)，提升张量核心并行度，腰斩显存占用。
   * **纯 GPU 张量重塑**：重构自回归解码中生成的 token 输入流，采用 `topi.view(1, -1)` 替代低效的 CPU 列表循环与频繁的数据拷贝，完全打通了 GPU 内部计算通道。
2. **去过拟合与去偏差**：
   * **固定随机种子打乱（Seed-Shuffle）**：在切分训练集与验证集（95%/5%）前，使用固定随机种子 `42` 对样本全局打乱，消除了因时序话题分布不均引发的验证损失发散，保障了评估中立性与可重复性。
   * **计划采样（Scheduled Sampling）**：在训练的 Epoch 大循环中引入渐进式导师迫导衰减——在第 `0~30` 个 Epoch 内将 `teacher_forcing_ratio` 从 `0.8` 线性下调至 `0.1`，促使模型逐步过渡到完全自主生成状态，彻底解决了暴露偏差（Exposure Bias）造成的验证 Loss 恶性上升。

## 数据集

[小黄鸡聊天语料](https://github.com/candlewill/Dialog_Corpus)
[PTT中文語料](https://github.com/zake7749/Gossiping-Chinese-Corpus)

## 用法

### 运行测试
执行单元测试以验证代码和环境正常：
```bash
$ uv run pytest
```

### 数据预处理
提取训练数据。可以通过 `--corpus`（或 `-c`）指定语料，可选值为 `xhj` (小黄鸡) 和 `ptt` (PTT中文语料)，默认为 `xhj`：
```bash
# 处理小黄鸡语料（默认）
$ uv run pre_process.py

# 处理 PTT 语料
$ uv run pre_process.py --corpus ptt
```

### 训练
开始训练模型。同样可以使用 `--corpus` 或 `-c` 参数指定语料，默认为 `xhj`：
```bash
# 训练小黄鸡语料（默认）
$ uv run train.py

# 训练 PTT 语料
$ uv run train.py --corpus ptt
```

### 可视化训练过程
可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 人机交互对话
运行以下命令即可启动终端交互，与你训练好的模型进行实时聊天：
```bash
$ uv run chat.py
```

不正经的语料只能训练不正经的聊天机器人 ╮（﹀_﹀）╭

![image](https://github.com/foamliu/Chatbot/raw/master/images/result-1.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-2.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-3.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-4.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-5.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-6.jpg)
