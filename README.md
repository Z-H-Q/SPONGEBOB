# SpongeBob-Pro

SpongeBob-Pro 是一个约 **0.1B（≈8700 万）参数**的中英双语大语言模型（LLM）**全流程训练与评测代码库**。它用"麻雀虽小、五脏俱全"的方式，在一套代码里覆盖了从零训练一个"能对话的模型"所需的全部环节：

```
自训练 BPE 词表 → 数据预处理 → 预训练(Pretrain) → 指令微调(SFT) → 强化学习(GRPO) → 对话 / 评测
```

## ✨ 特性

- **完整管线**：Tokenizer 训练 → 文本预训练 → SFT 指令微调 → GRPO 强化学习 → 交互式对话，五步闭环
- **小模型大架构**：decoder-only Transformer，采用现代 LLM 的主流组件（RMSNorm、RoPE、GQA、SwiGLU、Flash Attention、权重绑定）
- **自训练 BPE Tokenizer**：15k 词表，中英混合，15 个预置特殊 token，自带 chat 模板
- **预训练数据预处理**：多进程流式 tokenize，`.jsonl → .bin/.meta` 二进制格式，训练时内存映射（memmap）零拷贝加载
- **SFT 只对回答算 loss**：多轮对话支持，user 部分全部 mask，仅 assistant 部分参与梯度
- **GRPO 强化学习**：格式奖励 + DeepSeek Judge 三维打分（fluency / factuality / instruction_following），组内归一化优势函数，k3 估计器 KL 约束
- **两套评测**：预训练用 C3 / XCOPA 多选题（困惑度比较）；微调后用 mini_bench 生成式评测（异步 DeepSeek Judge，不阻塞训练）
- **分布式友好**：预训练 / SFT / GRPO 均支持 DDP 多卡（torchrun 启动），并提供单卡版 `pretrain_without_ddp.py`（与 DDP 版逐行标注差异，便于学习）
- **SwanLab 实验追踪**：loss / lr / 评测指标 / GRPO 奖励曲线自动上报

---

## 目录

1. [项目定位](#1-项目定位)
2. [训练管线总览](#2-训练管线总览)
3. [模型架构](#3-模型架构)
4. [环境安装](#4-环境安装)
5. [仓库结构](#5-仓库结构)
6. [快速开始](#6-快速开始)
7. [数据格式](#7-数据格式)
8. [脚本与参数详解](#8-脚本与参数详解)
9. [GRPO 实现细节](#9-grpo-实现细节)
10. [评测体系](#10-评测体系)
11. [Checkpoint 与权重](#11-checkpoint-与权重)
12. [DDP 多卡训练](#12-ddp-多卡训练)
13. [已知问题与注意事项](#13-已知问题与注意事项)
14. [常见问题 FAQ](#14-常见问题-faq)
15. [许可证](#15-许可证)

---

## 1. 项目定位

- **目标**：在个人可承受的算力下（单卡即可跑通），完整经历 LLM 训练的每一个阶段，理解"模型是怎么被一步步训出对话能力的"。代码结构刻意保持简单、注释充分，适合学习、教学与二次开发。
- **模型**：`SpongeBobForCausalLM`，decoder-only 因果语言模型，默认 0.1B 配置约 **87,017,472 个参数**（87M）。
- **数据**：中英混合文本。预训练数据预处理时拼接全部文本再切成定长 chunks，词表为仓库内自训练的 15000 词 BPE tokenizer。
- **训练三阶段**：Pretrain（续写）→ SFT（对话）→ GRPO（思考格式 + 答案质量强化）。

## 2. 训练管线总览

```text
① 词表：文本 jsonl ──train_tokenizer.py──▶ tokenizer_15k/（15k BPE，含对话特殊 token）

② 预训练：文本 jsonl ──preprocess_data.py──▶ xxx.bin + xxx.meta
                                                       │
                              pretrain.py / pretrain_without_ddp.py（续写任务，全 token 算 loss）
                                                       │
                                                       ▼
                                            pretrain_768.pth
                                                       │
③ SFT：对话 jsonl ─────────────────── train_sft.py（只对 assistant 算 loss）──▶ sft_768.pth
                                                       │
                                                       ├──▶ eval.py 交互式对话
                                                       │
④ GRPO：mini_bench prompts ── train_grpo.py ──▶ grpo_768.pth
            （采样多条回答 → 格式检查 + DeepSeek Judge 打分 → 组内优势 → 策略梯度更新）

⑤ 评测：C3/XCOPA 多选题（困惑度）· mini_bench 生成式评测（DeepSeek Judge 三维打分）
```

每个阶段的产出都是下一阶段的输入，权重文件以 **fp16 裸 `.pth`** 形式在阶段间传递（详见[第 11 节](#11-checkpoint-与权重)）。

## 3. 模型架构

模型实现在 [`model/model_spongebob_pro.py`](model/model_spongebob_pro.py)，配置类在 [`model/config.py`](model/config.py)。

### 3.1 架构组件

| 组件 | 采用方案 | 说明 |
| --- | --- | --- |
| 主干 | Decoder-only Transformer | 12 层 `SpongeBobBlock`，自回归因果语言建模 |
| 归一化 | **RMSNorm**（Pre-Norm） | 先归一化再进入注意力 / FFN，训练更稳定；无 bias |
| 位置编码 | **RoPE**（旋转位置编码） | base = 10000，按 `max_position_embeddings`（默认 32768）预计算 cos/sin 频率并注册为 buffer（不随训练更新） |
| 注意力 | **MHA + GQA** | 12 个 query 头 / 4 个 KV 头，KV 头通过 `repeat_kv` 复制对齐，节省推理显存 |
| 注意力加速 | **Flash Attention（SDPA）** | PyTorch ≥ 2.0 时走 `F.scaled_dot_product_attention`（无 padding 走 `is_causal=True` 最快路径，有 padding 合成 causal+padding mask）；训练、无 KV cache 时启用；否则回退传统 softmax 实现 |
| FFN | **SwiGLU** | `silu(gate(x)) * up(x) → down`，中间维度 2048（≈ hidden × 8/3 再对齐 64） |
| 输出层 | **权重绑定** | `lm_head.weight = embed_tokens.weight`，省掉约 1152 万参数 |
| 推理加速 | **KV Cache** | `use_cache` + `past_key_values` 增量解码；支持 `logits_to_keep` 只算最后若干 token 的 logits |

### 3.2 默认 0.1B 配置（`SpongeBobConfig`）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 12 | Transformer 层数 |
| `num_attention_heads` | 12 | Query 注意力头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA） |
| `intermediate_size` | 2048 | FFN 中间维度 |
| `vocab_size` | 15000 | 词表大小（与 tokenizer 一致） |
| `max_position_embeddings` | 32768 | 预计算 RoPE 的最大长度 |
| `rope_theta` | 10000.0 | RoPE 基础频率 |
| `hidden_act` | silu | 激活函数 |
| `dropout` | 0.0 | 丢弃率 |
| `rms_norm_eps` | 1e-5 | RMSNorm 的 epsilon |
| `bos_token_id` / `eos_token_id` | 1 / 2 | `<|im_start|>` / `<|im_end|>` |
| `flash_attn` | True | 启用 SDPA 加速 |

> 按该配置：embedding 1152 万 + 每层约 629 万 × 12 层 ≈ **87,017,472（约 8700 万，即俗称 0.1B）**。通过 `--hidden_size` / `--num_hidden_layers` 可随时缩放模型尺寸（训练、保存、推理三处必须保持一致）。

### 3.3 特殊 Token（15 个，ID 0–14 固定）

| ID | Token | 用途 |
| --- | --- | --- |
| 0 | `<\|endoftext\|>` | 文本结束 / pad / unk |
| 1 | `<\|im_start\|>` | 对话开始（bos） |
| 2 | `<\|im_end\|>` | 对话轮次结束（eos） |
| 3 | `<think>` | 思考块开始（GRPO 阶段） |
| 4 | `</think>` | 思考块结束（GRPO 阶段） |
| 5 | `<pad>` | 显式 padding |
| 6 | `<unk>` | 未登录词 |
| 7 | `<\|system\|>` | 系统消息标记 |
| 8 | `<\|user\|>` | 用户消息标记 |
| 9 | `<\|assistant\|>` | 助手消息标记 |
| 10–11 | `<tool_call>` / `</tool_call>` | 工具调用（预留） |
| 12–13 | `<function>` / `</function>` | 函数调用（预留） |
| 14 | `<unused_0>` | 预留 |

对话格式遵循 `ChatML` 风格，例如：

```text
<|im_start|><|user|>你好，你是谁？<|im_end|><|assistant|>我是 SpongeBob！<|im_end|>
```

`tokenizer_config.json` 内置了 chat 模板（`apply_chat_template` 可直接用），但注意：**当前模板只渲染 `user` / `assistant` 两个角色**，`system` 消息不会被拼进 prompt。

## 4. 环境安装

- Python ≥ 3.9（开发环境为 3.11 / 3.12）
- **PyTorch ≥ 2.1**（模型使用 `torch.amp` 与 `F.scaled_dot_product_attention`；低于 2.0 只能走传统注意力，但仍可运行）
- CUDA 可选：训练脚本在无 GPU 时自动回退 CPU（`--device cpu`），GRPO / mini_bench 的 Judge 需要能访问 DeepSeek API 的网络

```bash
pip install torch torchvision            # 按你的 CUDA 版本从 pytorch.org 选装
pip install transformers datasets tokenizers numpy tqdm
pip install openai                       # Judge 评测（mini_bench / GRPO 奖励）需要
pip install swanlab                      # 可选：实验追踪（也可 --use_swanlab 0 关掉）
```

> 说明：`transformers` 仅用于 `PretrainedConfig` / `GenerationMixin` / `AutoTokenizer` 等基础组件，模型本体是手写的，不依赖任何现成模型权重。

## 5. 仓库结构

```text
SpongeBob-Pro/
├── model/
│   ├── config.py                  # SpongeBobConfig：模型超参配置类
│   ├── model_spongebob_pro.py     # 模型本体：RMSNorm/RoPE/GQA/SwiGLU/FlashAttn/KV Cache
│   └── __init__.py
├── train/
│   ├── train_tokenizer.py         # 训练 15k BPE tokenizer（中英混合）
│   ├── preprocess_data.py         # 见 dataset/（此处为旧入口，建议用 dataset/ 下同名脚本）
│   ├── pretrain.py                # 预训练（DDP 多卡版）
│   ├── pretrain_without_ddp.py    # 预训练（单卡版，与 DDP 版差异用 [DDP] 标注）
│   ├── train_sft.py               # SFT 指令微调（DDP，assistant-only loss）
│   ├── train_grpo.py              # GRPO 强化学习（格式奖励 + Judge 奖励）
│   └── utils.py                   # 学习率调度 / DDP 初始化 / 日志 / SkipBatchSampler
├── dataset/
│   ├── preprocess_data.py         # jsonl → .bin/.meta（多进程流式预处理）
│   ├── pretrain_dataset.py        # PretrainDataset：memmap 读取 .bin
│   ├── sft_dataset.py             # SFTDataset：对话拼接 + assistant-only label
│   └── grpo_dataset.py            # GRPODataset：mini_bench prompt 格式化
├── benchmark/
│   ├── evaluator.py               # C3 / XCOPA 多选题评测（困惑度比较法）
│   ├── clue_c3_eval_500.jsonl     # C3 中文阅读理解选择题（500 条）
│   ├── xcopa_zh_merged.jsonl      # XCOPA 中文因果推理题（600 条）
│   ├── test_xcopa_improved.py     # XCOPA 评测自检脚本（随机模型应≈50%）
│   └── mini_bench/
│       ├── eval.py                # run_inference + run_judge（同步/异步，DeepSeek Judge）
│       ├── run_test.py            # mini_bench 跑分示例入口
│       ├── 100miniSponge.jsonl    # 100 条中文评测 prompt
│       └── 200_test_shuffled.jsonl# 200 条洗牌后的测试 prompt
├── tokenizer_15k/                 # 已训练好的 tokenizer（可直接使用）
├── data/pretrain_data/            # 预处理产物示例（仅 .meta；.bin 需自行生成）
├── out_grpo/exp_1/                # GRPO 训练产物示例（rollout 日志）
└── eval.py                        # 交互式对话推理（单轮/多轮，流式输出）
```

## 6. 快速开始

下面是一条最小可跑通的全流程路径（命令均在**仓库根目录**下执行）。

### 6.0 获取仓库

```bash
git clone https://github.com/Z-H-Q/SPONGEBOB.git
cd SPONGEBOB   # 或你本地克隆后的目录名
```

> 仓库内已自带训练好的 `tokenizer_15k/`，第 ① 步可跳过；只有当你换了训练语料、想重新训练词表时才需要跑。

### 6.1 数据预处理（生成 .bin/.meta）

预训练输入是每行一条 JSON 的文本 jsonl（格式见[第 7.1 节](#71-预训练数据)）：

```bash
python dataset/preprocess_data.py \
  --input data_raw/pretrain.jsonl \
  --output data/pretrain_data/spongebob_pretrain_512 \
  --tokenizer ./tokenizer_15k \
  --seq_len 512
```

输出两个文件：

- `spongebob_pretrain_512.bin`：`uint16` 数组，形状 `[num_chunks, 512]`
- `spongebob_pretrain_512.meta`：元信息（seq_len / num_chunks / shape 等），训练时会校验

### 6.2 预训练（单卡）

```bash
python train/pretrain_without_ddp.py \
  --data_path ./data/pretrain_data/spongebob_pretrain_512.bin \
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

> `--max_seq_len` 必须与 `.meta` 里的 `seq_len` 完全一致（代码里有 assert 校验）。多卡改用 `train/pretrain.py` + `torchrun`（见[第 12 节](#12-ddp-多卡训练)）。

### 6.3 SFT 指令微调

```bash
python train/train_sft.py \
  --data_path ./data_raw/sft_train.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --save_dir ./out_sft \
  --save_weight sft \
  --epochs 2 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --max_seq_len 512 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --from_weight ./pretrain_out/h768_l12_bs128_lr0.001/global_step_xxx/pretrain_768.pth \
  --use_swanlab 0
```

### 6.4 对话体验

```bash
python eval.py \
  --model_path ./out_sft/h768_l12_bs128_lr2e-05/global_step_xxx/sft_768.pth \
  --tokenizer_path ./tokenizer_15k \
  --multi_turn
```

- `--model_type sft`：对话模式（`--multi_turn` 决定是否保留历史；`eval.py` 也会按文件名自动判断类型）
- `--model_type pretrain`：文本续写模式（只吃当前输入，不保留历史）
- 输入 `exit` / `quit` / `退出` 结束

### 6.5 GRPO 强化学习（需要 DeepSeek API Key）

```bash
python train/train_grpo.py \
  --data_path ./benchmark/mini_bench/100miniSponge.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --sft_model_path ./out_sft/h768_l12_bs128_lr2e-05/global_step_xxx/sft_768.pth \
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
  --judge_model deepseek-chat
```

## 7. 数据格式

### 7.1 预训练数据

- **Tokenize 训练输入**（`train_tokenizer.py` 与预处理共用）：每行 `{"text": "..."}`
- **预处理输出**：`.bin`（`uint16` 定长序列数组）+ `.meta`（元信息 JSON）
- **预处理逻辑**：逐行 tokenize → 每条末尾追加 `eos_token_id`（=2）→ 所有 token 首尾相接 → 切成定长 `seq_len` 的 chunks（尾部不足一条的丢弃）
- **训练时**：`PretrainDataset` 用 `np.memmap` 映射 `.bin`（不占内存），`input_ids` 与 `labels` 为同一序列，shift 在模型内部完成

> 仓库附带的 `.meta` 示例来自作者的真实预训练语料：约 473 万条文本、13.46 亿 token、263 万个 512 长度 chunks。`.bin` 本身未入库（太大），请自行生成。

### 7.2 SFT 数据

每行一条多轮对话（`datasets` 库自动 mmap，可放心放百万级文件）：

```json
{"conversations": [
  {"role": "user", "content": "请介绍一下你自己。"},
  {"role": "assistant", "content": "我是 SpongeBob，一个 0.1B 的中英双语模型！"},
  {"role": "user", "content": "你能做什么？"},
  {"role": "assistant", "content": "我可以聊天、续写文本。"}
]}
```

- 拼接格式：`<|im_start|><|user|>内容<|im_end|><|assistant|>内容<|im_end|>...`
- **Labels**：user 段（含 `<|user|>`、内容、`<|im_end|>`）全部置 `-100` 不参与 loss；assistant 段（含 `<|assistant|>` 标记）全部算 loss；padding 部分 `-100`
- 不足 `max_length` 用 `pad_id` 补齐，超出截断

### 7.3 GRPO / mini_bench 数据

每行一条 prompt（mini_bench 格式，**只有问题没有标准答案**，靠 Judge 打分）：

```json
{"id": 1, "category": "开放问答", "prompt": "我饿了怎么办？"}
```

`GRPODataset` 会格式化为：

```text
<|im_start|><|user|>我饿了怎么办？<|im_end|><|assistant|>
```

**Benchmark 数据集格式**：

```json
// C3（clue_c3_eval_500.jsonl）：中文阅读理解
{"id": 2040, "context": ["我公司现招聘一名经济法方面的律师……"], "question": "应聘这个工作的人必须：",
 "choice": ["超过35岁", "有管理经验", "会说普通话", "工作三年以上"], "answer": "工作三年以上"}

// XCOPA（xcopa_zh_merged.jsonl）：中文因果推理
{"premise": "那人打开水龙头。", "choice1": "厕所里满是水。", "choice2": "水从水龙头喷口流出。",
 "question": "effect", "label": 1}
```

## 8. 脚本与参数详解

### 8.1 三个训练脚本共有的参数

| 参数 | pretrain 默认 | SFT 默认 | GRPO 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `--save_dir` | `../pretrain_out/exp_mini` | `../out_sft/exp_1` | `../out_grpo/exp_1` | 保存根目录（其下按 `h{hidden}_l{layers}_bs{bs}_lr{lr}` 建子目录） |
| `--save_weight` | `pretrain` | `sft` | `grpo` | 权重文件名前缀 |
| `--epochs` | 2 | 2 | 900 | 训练轮数（GRPO 默认很大，实际用小值跑通即可） |
| `--batch_size` | 128 | 128 | 16 | 每卡 batch size |
| `--learning_rate` | 1e-3 | 2e-5 | 5e-7 | 初始学习率（SFT 建议 1e-5~1e-4；GRPO 建议 1e-6 量级） |
| `--device` | 自动 cuda:0 / cpu | 同左 | 同左 | 训练设备 |
| `--dtype` | bfloat16 | bfloat16 | bfloat16 | 混合精度类型（float16 时启用 GradScaler） |
| `--num_workers` | 8 | 8 | 8 | DataLoader 线程数 |
| `--accumulation_steps` | 1 | 1 | 1 | 梯度累积步数 |
| `--grad_clip` | 1.0 | 1.0 | 0.2 | 梯度裁剪阈值 |
| `--log_interval` | 10 | 10 | 1 | 日志打印间隔 |
| `--save_interval` | 3000 | 1000 | 20 | checkpoint 保存间隔 |
| `--hidden_size` | 768 | 768 | 768 | 隐藏层维度（务必与权重一致） |
| `--num_hidden_layers` | 12 | 12 | 12 | 层数（务必与权重一致） |
| `--max_seq_len` | 512 | 512 | 128 | 序列长度（GRPO 中只截 prompt） |
| `--from_resume` | 0 | 0 | 0 | 自动检测 save_dir 下最新 `resume.pth` 续训 |
| `--use_swanlab` | 1 | 1 | 1 | SwanLab 实验追踪开关 |
| `--swanlab_project` | SpongeBob-Pretrain | SpongeBob-SFT | SpongeBob-GRPO | SwanLab 项目名 |
| `--use_compile` | 1 | 1 | 1 | `torch.compile` 加速（出问题可关） |

### 8.2 `train/pretrain.py`（DDP 版）专属

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_path` | 空 | `.bin` 路径（不带扩展名会自动补） |
| `--from_weight` | `none` | 续训起点；注意这里走 `from_pretrained`（HF 格式目录），与 SFT 的裸 `.pth` 加载不同 |
| `--eval_bench` | 1 | 预训练中跑 C3/XCOPA 评测 |
| `--eval_interval` | 1000 | 评测间隔步数 |

`pretrain_without_ddp.py` 参数相同、逻辑一致，只是去掉 DDP；两文件用 `# [DDP]` 注释标出了全部差异，非常适合对照学习分布式改造。

### 8.3 `train/train_sft.py` 专属

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_path` | 空 | SFT 对话 jsonl |
| `--tokenizer_path` | `../tokenizer_15k` | tokenizer（SFTDataset 一开始就需要） |
| `--from_weight` | 空 | 预训练权重（裸 `.pth`，`load_state_dict` 加载，`strict=False`） |
| `--enable_eval` | 0 | 训练中周期性 mini_bench 生成式评测 |
| `--eval_interval` | 1000 | 评测间隔（0=关闭） |
| `--judge_api_key` | 空 | DeepSeek API Key（也可从环境变量 `DEEPSEEK_API_KEY` 读） |
| `--judge_model` | deepseek-chat | Judge 模型名 |

SFT 的 warmup 为总步数的 **10%**（比预训练的 3% 更长，更稳）。

### 8.4 `train/train_grpo.py` 专属

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_path` | 空 | mini_bench 格式 prompt jsonl |
| `--tokenizer_path` | `../tokenizer_15k` | tokenizer |
| `--sft_model_path` | （作者集群路径） | SFT 权重，同时初始化 policy 与 reference 模型 |
| `--max_gen_len` | 512 | 每个回复最多生成的 token 数 |
| `--num_generations` | 4 | 每条 prompt 采样的回复数（组内相对优势的分母） |
| `--beta` | 0.05 | KL 惩罚系数 |
| `--judge_api_key` / `--judge_model` | 空 / deepseek-chat | 奖励模型配置 |

### 8.5 `eval.py`（交互对话）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--model_path` | （作者集群路径） | `.pth` 权重路径；文件名含 `pretrain`/`sft` 会自动推断 `--model_type` |
| `--tokenizer_path` | `./tokenizer_15k` | tokenizer |
| `--model_type` | sft | `pretrain`（续写）/ `sft`（对话） |
| `--hidden_size` / `--num_hidden_layers` | 768 / 12 | 重建模型所需，与权重必须一致 |
| `--max_new_tokens` | 2048 | 单次回复最大长度 |
| `--temperature` / `--top_p` | 0.2 / 0.7 | 采样参数 |
| `--multi_turn` | 关 | 是否保留对话历史 |

### 8.6 `dataset/preprocess_data.py`

| 参数 | 含义 |
| --- | --- |
| `--input` | 输入 jsonl |
| `--output` | 输出前缀（生成 `.bin` 和 `.meta`） |
| `--tokenizer` | tokenizer 路径 |
| `--seq_len` | chunk 长度（默认 512） |
| `--num_workers` | 并行进程数（默认 CPU 核数，上限 32） |

内部实现：多进程 tokenize → 每 1000 万 token 落一次盘（防内存爆炸）→ `np.fromfile` 读取 → reshape 切块 → 写 `.bin`/`.meta`。

### 8.7 `train/train_tokenizer.py`

- 数据路径、输出目录、词表大小是文件顶部**写死的常量**，运行前必须修改
- `SPECIAL_TOKENS` 15 个（见[3.3 节](#33-特殊-token15-个id-014-固定)），训练后断言前 5 个 ID 正确
- 主流程里 `train_tokenizer(...)` 调用被注释，默认只跑 `eval_tokenizer`；加 `--test` 参数用前 10000 行试跑
- BPE 参数：`min_frequency=2`、ByteLevel 预分词、初始字母表 6000

## 9. GRPO 实现细节

GRPO（Group Relative Policy Optimization，分组相对策略优化）不用单独的价值网络，而是对**同一 prompt 的多条采样回复**做组内比较，用相对优势更新策略。本仓库实现如下：

### 9.1 采样（Rollout）

- policy 模型对每条 prompt 采样 `--num_generations` 条回复：`temperature=0.3`、`do_sample=True`、`repetition_penalty=1.2`
- 生成用左 padding；`pad_token` 不能等于 `eos_token`（否则左 padding 的 token 会立刻触发停止）
- 每步的 prompt / 回复 / 奖励 / 各维度打分都会写盘到 `<save_dir>/<run_name>/data_log/global_step_*.jsonl`，便于事后分析

### 9.2 奖励计算（格式 + Judge 两级）

```
回复文本
  │
  ├─ 格式检查：恰含 1 个 <think>、1 个 </think>，且整体匹配正则 ^<think>\n.*?\n</think>\n.+$
  │     └─ 不通过 → reward = 0（直接淘汰，不调 API）
  │
  └─ 通过 → 取出 </think> 之后的正文，并发（10 线程）调 DeepSeek Judge
           对 fluency / factuality / instruction_following 三个维度打 0/1 二值分
        → reward = 三维均值（∈ {0, 1/3, 2/3, 1}）；Judge 调用失败也算 0
```

- Judge 输出要求 JSON，解析容错（支持 ```` ```json ```` 围栏、正则兜底提取）
- Judge prompt 明确要求"如果无法判断，则视为不通过"（从严）

### 9.3 优势函数与 Loss

```text
组内归一化：A_i = (r_i - mean(组内 r)) / (std(组内 r) + 1e-4)，clip 到 [-10, 10]
          再对整个 batch 做一次全局标准化

KL 估计（k3 估计器）：kl = exp(r_ref - r_policy) - (r_ref - r_policy) - 1   （r 为 log-prob）

token 级 loss = -( exp(logp - logp.detach()) * A - beta * kl )   ← 只对第一个 EOS 之前的 token 求和
总 loss = 各序列按有效长度求均值 → 再跨 batch 求均值（可配合梯度累积）
```

- **Reference 模型**与 policy 同权重初始化、冻结、只用于 KL 估计
- 训练日志里 `Format / All / None` 分别对应：格式通过率、某 prompt 全部采样满分的比例（solve_all）、全部零分的比例（solve_none）

## 10. 评测体系

### 10.1 预训练 benchmark：C3 / XCOPA（多选题，困惑度比较）

`benchmark/evaluator.py`，不需要 API，纯本地：

- **方法**：把 context 分别与每个选项拼接，计算**选项部分**的平均负对数似然（loss），取 loss 最低的选项为预测
- **C3**（500 条）：中文阅读理解选择题，`context(拼接) + question + choice[] + answer`
- **XCOPA**（600 条）：中文因果推理二选一；`question == "cause"` 时 prompt 为 `"{premise}这是因为："`，`"effect"` 时为 `"{premise}所以："`
- 训练脚本每 `--eval_interval` 步跑一次并上报 SwanLab（`c3_accuracy` / `xcopa_accuracy`）
- `benchmark/test_xcopa_improved.py`：用随机初始化的 256 维小模型自检评测代码——准确率应接近 50% 随机基线

### 10.2 mini_bench：生成式评测（DeepSeek Judge）

`benchmark/mini_bench/eval.py` 提供两个通用函数：

- `run_inference(model, tokenizer, num_samples=3, max_prompts=None)`：对 `100miniSponge.jsonl` 的每条 prompt 采样 `num_samples` 条回复，返回 `(prompt, responses, responses_with_special)` 三元组
- `run_judge(pairs, ...)`：并发调 DeepSeek Judge，输出每维度 `{dim}_avg{n}`（平均分）与 `{dim}_pass{n}`（至少一条采样的通过率）以及 `mean_avg{n}` / `mean_pass{n}`
- `run_judge_async(...)`：把 Judge 放到后台线程，**训练不阻塞**，结果写 jsonl 并自动上报 SwanLab（SFT 训练中周期性评测就用它）

环境变量：`export DEEPSEEK_API_KEY="..."`（也可用 `--judge_api_key` 传入）。

快速跑分示例（`benchmark/mini_bench/run_test.py`，需先改脚本里的 `WEIGHT` 路径）：

```bash
export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
python benchmark/mini_bench/run_test.py --max_prompts 5
```

## 11. Checkpoint 与权重

每个训练脚本按 `--save_dir/<run_name>/global_step_{N}/` 保存两个文件：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `{save_weight}_{hidden_size}.pth` | 纯模型权重 state_dict，**fp16** 存储 | 传给下一阶段 / eval.py 推理 |
| `resume.pth` | 模型 + 优化器 + GradScaler + epoch/step/global_step + swanlab_id | `--from_resume 1` 断点续训 |

另外：

- SFT 训练中开启 `--enable_eval` 时，评测样本写到 `<save_dir>/<run_name>/valid_samples/global_step_*.jsonl`
- GRPO 每次采样的 rollout 写到 `<save_dir>/<run_name>/data_log/global_step_*.jsonl`

**权重传递约定**：由于保存的是"裸 state_dict + fp16"，加载方必须自己用一致的 `hidden_size` / `num_hidden_layers` 重建模型再 `load_state_dict`（`eval.py`、`train_sft.py`、`train_grpo.py` 都是这么做的）。注意 `train/pretrain.py` 的 `--from_weight` 走的是 `from_pretrained`（要求 HF 格式目录、含 config.json），与 SFT/GRPO 的裸 `.pth` 加载方式**不同**。

## 12. DDP 多卡训练

三个训练脚本都通过 `init_distributed_mode()` 读环境变量判断是否多卡（`RANK == -1` 即单机单卡模式），因此直接用 `torchrun` 启动即可：

```bash
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
  --data_path ./data_raw/sft_train.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight ./pretrain_out/.../pretrain_768.pth \
  --save_dir ./out_sft
```

内部机制：`DistributedSampler` 分片数据 → DDP 包模型（`freqs_cos`/`freqs_sin` buffer 忽略同步）→ 总步数除以 `world_size` → 只有主进程（rank 0）保存 checkpoint / 跑评测 / 上报 SwanLab。单卡训练直接 `python train/xxx.py`，无需任何改动。

> 提示：Windows 上多进程训练脚本内若未加 `if __name__ == "__main__":` 保护会报错——本仓库三个训练脚本的入口均在其保护内，可以直接跑。

## 13. 已知问题与注意事项

按"踩坑概率"排序，动手前请通读：

1. **硬编码路径**：`train_tokenizer.py` 的 `DATA_PATH/TOKENIZER_DIR`、`eval.py` 的 `--model_path` 默认值、`benchmark/mini_bench/run_test.py` 的 `WEIGHT`、`benchmark/test_xcopa_improved.py` 中的路径，都是作者训练集群（`/apdcephfs_qy4/...`）的绝对路径，本地运行前必须改成自己的路径。
2. **预训练 benchmark 的 import 与实际目录不匹配**：`train/pretrain.py` 顶部 `from benchmark.pretrain.evaluator import run_benchmark`，且 `_BENCH_PRETRAIN_DIR` 指向 `benchmark/pretrain/`；但仓库实际布局是 `benchmark/evaluator.py` 与 `benchmark/*.jsonl` 在根目录。也就是说 **DDP 版预训练脚本目前开箱会 ImportError**。二选一修复：① 建 `benchmark/pretrain/` 目录，放入 `__init__.py`、`evaluator.py` 和两个 jsonl；② 把 import 改成 `from benchmark.evaluator import run_benchmark`（`pretrain_without_ddp.py` 用的就是正确路径，可直接照抄）并改 `_BENCH_PRETRAIN_DIR`。另外该脚本里 benchmark 用的 tokenizer 是从空字符串加载的（`AutoTokenizer.from_pretrained('')`），需改为 `./tokenizer_15k`。
3. **SwanLab API Key 硬编码**：三个训练脚本内 `swanlab.login(api_key=...)` 写死了作者的 key。使用前换成自己的（或设 `--use_swanlab 0` 关闭），**不要把 key 提交到公开仓库**。
4. **权重是 fp16 保存的**：加载时必须用一致的 `hidden_size / num_hidden_layers` 重建模型，否则 `load_state_dict` 会 shape 不匹配。
5. **`--max_seq_len` 必须与 `.meta` 一致**：`PretrainDataset` 里有 assert，不一致直接报错。
6. **GRPO 的 `--sft_model_path` 默认值指向作者本地路径**，务必显式传入你自己的 SFT 权重；加载用 `strict=False`，不匹配的键会静默跳过。
7. **GRPO / mini_bench 生成依赖左 padding**，且 `pad_token` 不能等于 `eos_token`（否则左 padding 立即触发 EOS、生成空回复）。代码内已处理（pad 用 unk 或 endoftext），自己改生成逻辑时注意保留。
8. **chat 模板不渲染 system 角色**：虽然词表里有 `<|system|>` token，但 `tokenizer_config.json` 的 chat_template 只处理 user / assistant，system 消息会被忽略。
9. **脚本运行方式**：推荐 `python train/xxx.py`（脚本内部通过 `__package__` + `sys.path` 处理了相对导入）；不要直接 `python -m` 或从别的目录执行。
10. **GRPO 的 Judge 是网络调用**：每条格式正确的回复都会调 DeepSeek API（10 线程并发），训练步进会被 API 延迟拖慢；没钱/没网时可用 `--epochs` 小值验证管线，或只跑 `--num_generations 2`。
11. **`torch.compile` 与 DDP 同时开启时**，checkpoint 恢复走 `_orig_mod` 解包路径；若遇恢复异常可先 `--use_compile 0` 排查。
12. **数据目录只带 `.meta`**：`data/pretrain_data/` 里只有元信息示例，`.bin` 需自己跑预处理生成。

## 14. 常见问题 FAQ

**Q：模型到底多少参数？**
默认 0.1B 配置共 87,017,472 参数（约 8700 万），习惯上四舍五入称 0.1B。改 `--hidden_size` / `--num_hidden_layers` 即可缩放。

**Q：没有 GPU 能跑吗？**
能。训练脚本无 CUDA 时自动回退 CPU（加 `--device cpu`），但速度慢得多；GRPO / mini_bench 的 Judge 需要网络访问 DeepSeek API。

**Q：为什么权重存成 fp16？**
省一半磁盘/带宽。加载时先按 config 重建 fp32 模型再 `load_state_dict`，PyTorch 会自动做 dtype 转换（`eval.py` 就是这么做的）。

**Q：为什么保存裸 `.pth` 而不是 HuggingFace 格式？**
设计取舍：省去 config.json 的维护，靠"约定 hidden_size / num_hidden_layers 一致"传递信息。代价是换人使用时容易忘带参数，所以本 README 反复强调这一点。

**Q：GRPO 为什么要 `<think>` 格式？**
格式奖励是最便宜的奖励信号（零 API 成本、天然可验证）。先让模型学会"先想后答"的格式，再用 Judge 压答案质量，形成两级奖励。

**Q：SFT 的 loss 为什么只算 assistant 部分？**
语言建模天然会预测所有 token，但 user 的内容不是模型该"复述"的目标；mask 掉 user 段（置 -100 被 `CrossEntropyLoss` 忽略）后，梯度只来自"模型自己该说的话"。

## 15. 许可证

本项目采用 **MIT License** 开源，完整文本见 [`LICENSE`](LICENSE)。你可以自由使用、修改、分发，但需保留原始版权与许可证声明。

---

### 致谢

- 模型架构参考 Llama 系列（RMSNorm / RoPE / GQA / SwiGLU）
- GRPO 算法参考 DeepSeekMath 的 Group Relative Policy Optimization 及 DeepSeek-R1 的训练思路
- 评测 Judge 使用 DeepSeek API
- 实验追踪使用 SwanLab
