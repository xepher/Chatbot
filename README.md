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
提取训练数据：
```bash
$ uv run extract.py
$ uv run pre_process.py
```

### 训练
```bash
$ uv run train.py
```

### 可视化训练过程
可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示

```bash
$ uv run demo.py
```

不正经的语料只能训练不正经的聊天机器人 ╮（﹀_﹀）╭

![image](https://github.com/foamliu/Chatbot/raw/master/images/result-1.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-2.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-3.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-4.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-5.jpg)
![image](https://github.com/foamliu/Chatbot/raw/master/images/result-6.jpg)
