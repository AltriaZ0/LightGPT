from model_exec_config import *
from model import LanguageModel
import torch
import torch.nn.functional as F
import random
from nvtx import nvtx_range
import os, json, pickle, glob
from data import MemmapTokenDataset  # 你之前的 data.py 里已经定义

def _resolve_path(base_dir, p):
    """把 meta.json 内的相对 bin 路径解析成绝对路径；若已是绝对路径原样返回"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))

def load_vocab_and_val_dataset(cache_dir="./data_cache", meta_path=None):
    """
    从 TextDataProcessor 的缓存三件套加载：
    - 词表: *.char_map.pkl -> char_to_idx, idx_to_char, vocab_size
    - 验证集: *.tokens.bin + *.meta.json -> MemmapTokenDataset 切片 (val)
    """
    cache_dir = os.path.abspath(cache_dir)

    # 1) 定位 meta.json
    if meta_path is None:
        metas = glob.glob(os.path.join(cache_dir, "*.meta.json"))
        if not metas:
            raise FileNotFoundError(f"在 {cache_dir} 未找到 *.meta.json")
        if len(metas) > 1:
            raise RuntimeError("发现多个 *.meta.json，请显式指定 meta_path：\n" + "\n".join(metas))
        meta_path = metas[0]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    # 2) 解析 bin_path
    bin_path = _resolve_path(meta_dir, os.path.basename(meta["bin_path"]))

    # 3) 定位 char_map.pkl（按 txt_name 派生；兜底通配）
    txt_basename = os.path.basename(meta.get("txt_name", ""))  # e.g. FullNovel.txt
    if txt_basename:
        map_path = os.path.join(meta_dir, f"{txt_basename}.char_map.pkl")
    else:
        cands = glob.glob(os.path.join(meta_dir, "*.char_map.pkl"))
        if not cands:
            raise FileNotFoundError("未找到 *.char_map.pkl（且 meta.json 缺少 txt_name 字段）。")
        map_path = cands[0]

    with open(map_path, "rb") as f:
        maps = pickle.load(f)
    char_to_idx = maps["char_to_idx"]
    idx_to_char = maps["idx_to_char"]
    vocab_size  = maps["vocab_size"]

    # 4) 基于 meta 切出 val memmap 视图
    n_total   = int(meta["n_total"])
    split_idx = int(meta["split_idx"])
    dtype     = meta["dtype"]  # 'uint16' 或 'uint32'

    # 简单尺寸校验（可选）
    dtype_bytes = 2 if dtype == "uint16" else 4
    expected_bytes = n_total * dtype_bytes
    actual_bytes   = os.path.getsize(bin_path)
    if expected_bytes != actual_bytes:
        raise ValueError(f".tokens.bin 尺寸不匹配：expect={expected_bytes}, actual={actual_bytes}\n  {bin_path}")

    # val 数据集（不把整集读入内存）
    val_ds = MemmapTokenDataset(bin_path, split_idx, n_total - split_idx, dtype=dtype)

    return (char_to_idx, idx_to_char, vocab_size), val_ds, meta_path

# 推理专用处理器
class InferenceProcessor:
    def __init__(self, vocab_state):
        # 改：直接接收 (char_to_idx, idx_to_char, vocab_size)
        self.char_to_idx, self.idx_to_char, self.vocab_size = vocab_state

    @nvtx_range()
    def decode_text(self, indices):
        return ''.join(self.idx_to_char.get(int(i), '?') for i in indices)

    @nvtx_range()
    def encode_text(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]  # 0 作为 <unk>


@nvtx_range()
def generate_streaming(model, processor, context, max_new_tokens=100, temperature=1.0, top_k=None):
    """
    流式生成文本，逐字符输出
    :param model: 训练好的模型
    :param processor: 处理器，用于解码
    :param context: 起始上下文 (1, T)
    :param max_new_tokens: 生成多少个新 token
    :param temperature: 温度，控制随机性
    :param top_k: 是否启用 top-k 采样
    :return: 生成的完整 token 序列
    """
    generated_tokens = context.clone()  # (1, T)
    print("【模型输出】生成的文本: \n", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # 截取上下文窗口
            input_tokens = generated_tokens[:, -block_size:]  # (1, block_size)
            AMP_DTYPE = torch.bfloat16  # 或 torch.float16
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                # 前向传播
                logits, _ = model(input_tokens)  # (1, block_size, vocab_size)
            logits = logits[:, -1, :] / temperature  # 只取最后一个 token，应用温度


            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 拼接
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # 实时输出字符
            char = processor.decode_text(next_token[0].tolist())
            print(char, end="", flush=True)

    print()  # 换行
    return generated_tokens

@nvtx_range()
def output():
    # 标记一个代码段开始
    with torch.cuda.nvtx.range("加载处理器状态与验证集"):
        vocab_state, val_data, meta_path = load_vocab_and_val_dataset(cache_dir="data_cache", meta_path=None)
        processor = InferenceProcessor(vocab_state)
        vocab_size_from_cache = processor.vocab_size

    with torch.cuda.nvtx.range("Load Model"):
        # 用缓存的 vocab_size 实例化模型，超参仍来自 model_config
        model = LanguageModel(vocab_size_from_cache, block_size, Embedding_dim, num_heads, numOfLayers, dropout).cuda()
        model.eval()
        
    # 加载模型权重
    torch.cuda.nvtx.range_push("Load Weights")
    model.load_state_dict(torch.load('model_weights.pth', map_location='cuda'))
    print("模型加载完成")
    torch.cuda.nvtx.range_pop()


    torch.cuda.nvtx.range_push("用户交互")
    # 5. 用户选择模式
    # print("随机续写还是输入上文？(r: 随机续写, i: 输入上文)")
    # choice = input().strip().lower()

    # if choice == 'i':
    #     print("请输入上文:")
    #     input_text = input().strip()
    #     if not input_text:
    #         print("输入为空，使用默认起始符")
    #         input_text = "\n"
    #     input_tokens = processor.encode_text(input_text)
    #     if not input_tokens:
    #         print("输入无法编码，使用换行符作为起始")
    #         input_tokens = [processor.char_to_idx.get('\n', 0)]
    #     context = torch.tensor([input_tokens], dtype=torch.long).cuda()
    # else:

    # 随机选择一个起始位置
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)
    context = val_data[start_idx:start_idx + block_size].unsqueeze(0).cuda()  # (1, block_size)

    with torch.cuda.nvtx.range("打印上文"):
        # 6. 打印上文
        print("【模型输出】上文:", processor.decode_text(context[0].tolist()))
    torch.cuda.nvtx.range_pop()

    # 7. 流式生成
    generated_full = generate_streaming(
        model=model,
        processor=processor,
        context=context,
        max_new_tokens=max_new_tokens,
        temperature=0.8,   # 控制多样性
        top_k=50           # 提高生成质量
    )
    with torch.cuda.nvtx.range("对比真实后续"):
        # 8. 对比真实后续（仅在随机模式下有意义）
        # if choice == 'r':
        real_next = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens]
        print("【真实文本】实际后续: \n" + processor.decode_text(real_next.tolist()))


# 运行
if __name__ == "__main__":
    output()