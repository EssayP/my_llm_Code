# my_llm_Code — 本地 Python 环境

本仓库是一组 LLM / 深度学习相关的手写实现练习代码（Transformer、RLHF、Tokenizer、
梯度下降、解码策略等）。所有代码只依赖 **PyTorch** 和 **NumPy**。

## 环境信息

- Python：3.9.6（系统自带 `/usr/bin/python3`）
- 虚拟环境：项目根目录下的 `.venv/`
- 平台：macOS arm64（Apple Silicon），PyTorch 自带 **MPS** 加速
- 已安装核心包：
  - `torch==2.8.0`
  - `numpy==2.0.2`

## 一次性创建环境

```bash
cd /Users/wenjianping/school/my_llm_Code
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 日常使用

激活环境：

```bash
source .venv/bin/activate
```

退出环境：

```bash
deactivate
```

也可以不激活，直接用 `.venv/bin/python` 运行某个文件：

```bash
.venv/bin/python "transformer/MultiHeadAttention.py"
.venv/bin/python "RLHF/DPO.py"
```

## 验证环境

```bash
.venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__, torch.backends.mps.is_available())"
```

期望输出类似：`2.8.0 2.0.2 True`

## 验证结果

在当前环境下逐个 import 仓库里所有 `.py` 文件：

| 结果 | 文件 |
|------|------|
| ✅ OK | `Decoding/top-k.py` |
| ✅ OK | `Gradient Descent/BGD.py` |
| ✅ OK | `Gradient Descent/SGD.py` |
| ✅ OK | `RLHF/DPO.py` |
| ✅ OK | `RLHF/GRPO_loss.py` |
| ✅ OK | `RLHF/GSPO.py` |
| ✅ OK | `RLHF/PPO_interview.py` |
| ✅ OK | `RLHF/PPO_loss.py` |
| ✅ OK | `RLHF/PPO_normal.py` |
| ✅ OK | `RecModel/SequenceModel/DIN.py`（空文件） |
| ❌ FAIL | `Tokenizer/BPE.py`（**代码 bug**，非环境问题，见下） |
| ✅ OK | `transformer/MQA.py` |
| ✅ OK | `transformer/MultiHeadAttention.py` |
| ✅ OK | `transformer/Positional_embedding.py` |
| ✅ OK | `transformer/ScaledDotPorductAttention.py` |
| ✅ OK | `transformer/dropout.py` |
| ✅ OK | `transformer/selfAttention.py` |

环境本身完全可用，16/17 文件可直接 import。

## 源码中的小 bug（环境无关，按需自行修复）

以下都是代码本身的问题，跟环境无关，改成注释里那样即可：

1. `Tokenizer/BPE.py:1`
   `from collection import Counter` → `from collections import Counter`（少了个 `s`）
   同文件 `train_bpe` 没有 `return merges`，导致 `encode("lowest", merges)` 时 `merges` 为 `None`。

2. `Decoding/top-k.py:17`
   `logits.unsequeeze(0)` → `logits.unsqueeze(0)`（拼写错误）

3. `Gradient Descent/SGD.py:9`
   `for i in range(indices):` → `for i in indices:`（`indices` 已经是打乱后的下标数组）

4. `transformer/Positional_embedding.py:1`
   `from tkinter import Variable` 是误用，应为 `from torch.autograd import Variable`
   （实际上 `Variable` 在文件中并未使用，可直接删掉这行。）

5. `transformer/MultiHeadAttention.py:35`
   `score.masked_fill(mask == 0, float('inf'))` 逻辑应为 `float('-inf')`
   （padding 位置要被 mask 掉，应置 `-inf` 让 softmax 归零，而不是 `+inf`。）
