# SpongeBob-Pro

SpongeBob-Pro 是一个小规模“从零到可对话”的语言模型训练与评测代码库，包含：

- 自定义 tokenizer（BPE，15k 词表，中英文混合）
- 文本预训练（`.jsonl -> .bin/.meta`，支持 DDP）
- 指令微调 SFT（只对 assistant 部分计算 loss，多轮对话）
- 强化学习式 GRPO（基于输出格式校验 + DeepSeek Judge 三维打分）
- mini_bench 生成式评测（推理 + Judge，异步执行）
- 交互式推理脚本（单轮/多轮）

> 注：本 README 以仓库中的脚本实现为准；你需要保证模型尺寸（`hidden_size / num_hidden_layers`）和权重文件匹配。

## 目录结构快速了解

- `train/`
  - `train_tokenizer.py`：训练 BPE tokenizer（15k）
  - `pretrain.py`：预训练（支持 DDP）
  - `pretrain_without_ddp.py`：预训练（单卡/不使用 DDP）
  - `train_sft.py`：SFT（支持 DDP）
  - `train_grpo.py`：GRPO（支持 DDP）
  - `utils.py`：学习率调度、DDP 初始化、日志与采样器
- `dataset/`
  - `preprocess_data.py`：把预训练 jsonl 预处理为 `.bin/.meta`
  - `pretrain_dataset.py`：`.bin/.meta` 读取数据集
  - `sft_dataset.py`：SFT 对话 jsonl 读取与 assistant-only loss mask
  - `grpo_dataset.py`：GRPO prompt 读取（mini_bench 格式）
- `benchmark/`
  - `mini_bench/`：mini_bench 推理 + Judge
  - `evaluator.py`：C3/XCOPA 多选题评测逻辑（用于预训练 benchmark）
- `model/`
  - `config.py`：`SpongeBobConfig`
  - `model_spongebob_pro.py`：模型实现
- `eval.py`：交互式对话推理（加载 `.pth` 权重并生成）

## 环境与依赖

从代码可知主要依赖包括：`torch`、`transformers`、`datasets`、`tokenizers`；GRPO/mini_bench 的 Judge 需要 `openai` 并调用 DeepSeek API。

建议的安装方式（按你的 CUDA/版本自行调整）：

```bash
pip install torch transformers datasets tokenizers accelerate tqdm numpy
pip install openai
```

如果你需要 DDP（多卡），请使用带环境变量的启动方式（见下方“分布式训练”）。

## 快速开始：交互式推理

`eval.py` 会创建一个 `SpongeBobForCausalLM` 并加载 `.pth` 权重（注意要匹配 `hidden_size/num_hidden_layers`）。

```bash
python eval.py \
  --model_path /path/to/your_model.pth \
  --tokenizer_path ./tokenizer_15k \
  --model_type sft \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --multi_turn
```

- `--model_type`
  - `pretrain`：文本续写（每次只用当前输入，不保留历史）
  - `sft`：对话（是否保留历史由 `--multi_turn` 决定）
- 对话输入：输入 `exit/quit` 退出

## 分布式训练（DDP）

训练脚本通过 `init_distributed_mode()` 读取环境变量（如 `RANK/LOCAL_RANK`），因此你应使用 `torchrun` 启动 DDP，例如：

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS train/train_sft.py \
  --data_path /path/to/train.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight /path/to/pretrain_or_sft.pth
```

单卡训练则直接：`python train/train_sft.py ...`

## 数据准备总览

### 1) tokenizer 训练数据（`train_tokenizer.py`）

`train_tokenizer.py` 读取的 jsonl 每行包含：

- `{"text": "你的文本..."}`

该脚本里路径与参数是写死的常量（`DATA_PATH / TOKENIZER_DIR / VOCAB_SIZE`），你需要在运行前修改这些变量。

### 2) 预训练数据（`pretrain.py`）需要 `.bin/.meta`

`train/pretrain.py` 依赖 `PretrainDataset`，它加载的是：

- `xxx.bin`（`uint16`，形状为 `[num_chunks, seq_len]`）
- `xxx.meta`（记录 `seq_len/num_chunks/shape` 等）

用 `dataset/preprocess_data.py` 从 jsonl 生成：

```bash
python dataset/preprocess_data.py \
  --input /path/to/pretrain.jsonl \
  --output /path/to/spongebob_pretrain_512 \
  --tokenizer ./tokenizer_15k \
  --seq_len 512
```

预处理逻辑：每行从 jsonl 读取 `text`，tokenize 后追加 `eos_token_id`，再拼接切分成定长 `seq_len` chunks，最终输出 `.bin/.meta`。

### 3) SFT 数据格式（`train_sft.py`）

`SFTDataset` 期望每行 JSON 为：

```json
{
  "conversations": [
    {"role": "user", "content": "问题/指令..."},
    {"role": "assistant", "content": "assistant 回复..."}
  ]
}
```

训练时只对 assistant 部分计算 loss（user token 会被 mask 为 `-100`），并会对话拼接为：

- `<|im_start|>` + 轮次内容 + `<|im_end|>` + `<|assistant|>` 标记等

### 4) GRPO 数据格式（`train_grpo.py`）

`GRPODataset` 从 jsonl 读取 mini_bench 形式的 prompt，每行 JSON 为：

```json
{"id": 1, "category": "开放问答", "prompt": "问题文本"}
```

训练脚本会把 prompt 格式化为：

`<|im_start|><|user|>{prompt}<|im_end|><|assistant|>`

## 训练流程

### 预训练（Pretrain）

使用 `train/pretrain.py`（DDP 支持）。核心参数：

- `--data_path`：预处理后的 `xxx.bin`（或不带扩展名也会自动补 `.bin`）
- `--save_dir`：保存 checkpoint 根目录
- `--hidden_size` / `--num_hidden_layers`：模型尺寸（要与权重/推理一致）
- `--max_seq_len`：数据集 `seq_len`（必须和 `.meta` 中一致）

示例（单卡）：

```bash
python train/pretrain.py \
  --data_path ./data/pretrain/spongebob_pretrain_512.bin \
  --save_dir ./pretrain_out \
  --save_weight pretrain \
  --epochs 2 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --max_seq_len 512 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --eval_bench 0
```

#### 预训练 benchmark（C3 / XCOPA）注意点

`train/pretrain.py` 里默认读取路径为：

- `benchmark/pretrain/clue_c3_eval_500.jsonl`
- `benchmark/pretrain/xcopa_zh_merged.jsonl`

但当前仓库里这两个文件位于 `benchmark/` 根目录下。你需要：

- 要么把 jsonl 移动到 `benchmark/pretrain/` 再开启 `--eval_bench 1`
- 要么修改 `train/pretrain.py` 中 `_BENCH_PRETRAIN_DIR` 的路径

### SFT（指令微调）

核心参数：

- `--data_path`：SFT 对话 jsonl 路径
- `--tokenizer_path`：必须提前准备 tokenizer（如 `./tokenizer_15k`）
- `--from_weight`：从现有 `.pth` 权重继续训练（可选）
- `--enable_eval 1`：开启 mini_bench + DeepSeek Judge 生成式评测（需要 `--judge_api_key`）

示例（单卡）：

```bash
python train/train_sft.py \
  --data_path ./dataset/sft_train.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --save_dir ./out_sft \
  --save_weight sft \
  --epochs 2 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --max_seq_len 512 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --from_weight ./pretrain_out/.../pretrain_768.pth \
  --enable_eval 0
```

如果你要开启评测：

```bash
python train/train_sft.py ... \
  --enable_eval 1 \
  --eval_interval 1000 \
  --judge_api_key "YOUR_DEEPSEEK_API_KEY" \
  --judge_model "deepseek-chat"
```

### GRPO（强化学习式 GRPO）

核心逻辑：

- 生成完成后会严格校验输出格式：
  - 必须出现且仅出现一次 `<think>` 和 `</think>`，且整体匹配：
    `^<think>\n.*?\n</think>\n.+$`
- 格式不通过：reward = 0
- 格式通过：调用 DeepSeek Judge，分别对 `fluency/factuality/instruction_following` 给 0/1，并取三指标均值作为 reward

核心参数：

- `--data_path`：GRPO prompt jsonl（mini_bench 格式）
- `--tokenizer_path`：tokenizer 路径
- `--sft_model_path`：用于初始化 policy 与 reference（通常来自 SFT 权重）
- `--judge_api_key`：DeepSeek Judge API Key
- `--num_generations`：每条 prompt 生成多少个 completion（用于分组优势计算）

示例（单卡）：

```bash
python train/train_grpo.py \
  --data_path ./benchmark/mini_bench/100miniSponge.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --sft_model_path ./out_sft/.../sft_768.pth \
  --save_dir ./out_grpo \
  --save_weight grpo \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 5e-7 \
  --max_seq_len 128 \
  --max_gen_len 512 \
  --num_generations 4 \
  --beta 0.05 \
  --judge_api_key "YOUR_DEEPSEEK_API_KEY" \
  --judge_model "deepseek-chat" \
  --use_compile 0
```

## mini_bench 评测（推理 + Judge）

`benchmark/mini_bench/eval.py` 提供通用函数：`run_inference()` 和 `run_judge()`。

你可以先设置环境变量：

```bash
$env:DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
```

仓库里也提供了一个示例脚本（需要你改 `WEIGHT` 路径）：

```bash
python benchmark/mini_bench/run_test.py --max_prompts 5
```

默认脚本会加载：

- `tokenizer_15k/`
- `WEIGHT`（写在脚本里）

并打印 `fluency/factuality/instruction_following/mean` 的 Judge 指标。

## 贡献与许可证

如果你在数据格式、评测脚本、训练稳定性方面做了改进，欢迎在仓库里提交 PR（或先讨论后再实现）。

本项目采用 **MIT License** 开源。

- 完整许可证文本见：`LICENSE`
- 你可以自由使用、修改、分发本项目代码（需保留原始版权与许可证声明）
