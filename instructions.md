# H100 上 MXFP4 量化 + LoRA 的推理效率测试（Rollout 场景）

> 目标：在 **H100** 上对 **Qwen2.5-3/7/14/32B（优先）** 与 **LLaMA-3.1-8B** 进行 **MXFP4 linear + LoRA** 配置的推理效率评测，模拟强化学习 **Rollout** 阶段的长输出场景。

---

## 测试设置

- **测试精度**：MXFP4（NVFP4 实现）
- **测试模型**：
  - Qwen2.5-3B / 7B / 14B / 32B（优先）
  - LLaMA-3.1-8B
- **模型格式**：MXFP4 Linear + LoRA  
  - LoRA rank：默认 32（后续可做 16 / 64 消融）
- **输入输出**：
  - input length = **1024**
  - output tokens = **4096 ~ 8192**
  - batch size = **1**
- **数据集**：`openai/gsm8k`（仅测速度也可用随机数据）
- **推理框架**：Transformers、vLLM、SGLang（本文示例以 **vLLM** 为主）
- **阶段**：模拟 **RL Rollout** 阶段

---

## 环境与版本隔离

由于**量化**与**推理**依赖的 `compressed_tensors` 版本不同，建议使用**两个独立的 Conda 环境**：

1) **量化环境：`llmcompressor`**（用于 NVFP4 量化）  
2) **推理环境：`vllm`**（用于 vLLM 测速）

---

## 一、量化为 NVFP4

```bash
# 1) 创建并激活量化环境
conda create -n llmcompressor python=3.12 -y
conda activate llmcompressor

# 2) 安装依赖
pip install -e .
pip install nvidia-ml-py

# 3) 将模型量化为 NVFP4（示例命令）
python test_scripts/quantize_nvfp4.py --model Qwen/Qwen2.5-3B
python test_scripts/quantize_nvfp4.py --model Qwen/Qwen2.5-7B
python test_scripts/quantize_nvfp4.py --model Qwen/Qwen2.5-14B
python test_scripts/quantize_nvfp4.py --model Qwen/Qwen2.5-32B
python test_scripts/quantize_nvfp4.py --model meta-llama/Llama-3.1-8B
```
量化完成后，记录各模型对应的 NVFP4 权重目录，例如：Qwen2.5-3B-NVFP4A16（下文将用作推理 --model 输入路径）。

## 二、使用 vLLM 进行推理与测速

```bash
# 1) 创建并激活推理环境
conda create -n vllm python=3.12 -y
conda activate vllm

# 2) 安装依赖
pip install vllm
pip install peft
pip install pandas
```
脚本说明
`test_scripts/efficiency_test.py`：对 `prefill` 与 `decode` 进行分别测速，覆盖长输出`（4096–8192）`与 `batch_size=1`、`batch_size>1` 的 RL Rollout 场景。
示例（以 Qwen/Qwen2.5-3B 为例）


1. 全精度（baseline）
```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model Qwen/Qwen2.5-3B \
  --batch_size 1
```

2. NVFP4 量化后的基座模型
```bash
# Qwen2.5-3B-NVFP4A16 为量化产物的本地路径（示例名）
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-3B-NVFP4A16 \
  --batch_size 1
```

3. NVFP4 + LoRA（bf16）
```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-3B-NVFP4A16 \
  --add_lora \
  --lora_path bunnycore/qwen-2.5-3b-lora_model \
  --batch_size 1
```

## 三、生成 Dummy LoRA（用于纯速度基准）
当仅关注 速度 而不关注 精度 时，可使用 零初始化的 `Dummy LoRA` 来模拟「加载 LoRA 分支」的开销。(同样在 vllm 环境中测试)

示例：为 `Qwen/Qwen2.5-7B` 生成 `Dummy LoRA（rank=64）`
```bash
python test_scripts/generate_dummy_lora.py \
  --base_model Qwen/Qwen2.5-7B \
  --lora_rank 64 \
  --lora_path /path/to/qwen25_7b_lora_init_default
```

随后在 efficiency_test.py 中将 --lora_path 替换为上述 Dummy LoRA 路径即可：

```bash
CUDA_VISIBLE_DEVICES=0 \
python test_scripts/efficiency_test.py \
  --model /path/to/Qwen2.5-7B-NVFP4A16 \
  --add_lora \
  --lora_path /path/to/qwen25_7b_lora_init_default \
  --batch_size 1
```
由于 VLLM 会根据 max_lora_rank 预分配显存已达到最佳的性能，我在 efficiency_test.py 默认 max_lora_rank为 16 （默认值），如果需要修改 lora rank 进行消融测试，请保证 efficiency_test 中的 max_lora_rank 大于当前传入的 lora rank