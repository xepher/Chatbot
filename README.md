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

### 演示
运行演示程序。同样支持 `--corpus` 传入（如果项目包含 demo.py）：
```bash
# 运行小黄鸡（默认）
$ uv run demo.py

# 运行 PTT
$ uv run demo.py --corpus ptt
```

不正经的语料只能训练不正经的聊天机器人 ╮（﹀_﹀）╭

![image](https://github.com/foamliu/Chatbot/raw/master/images/result-1.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-2.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-3.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-4.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-5.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-6.jpg)
