# === imports ===
import os, json, pickle, glob
import torch
from model import *
from model_train_config import *
from data import MemmapTokenDataset   # ← 你改造后的 data.py 里已有
# from data import TextDataProcessor  # 若你需要重建或再处理，可另外导入

# === 配置 ===
CACHE_DIR = "./data_cache"   # 放你 *.meta.json / *.char_map.pkl / *.tokens.bin 的目录
META_PATH = None             # 如果留空，会在 CACHE_DIR 下自动找唯一的 *.meta.json
BIN_PATH_OVERRIDE = None     # 若跨机导致 meta 里的 bin_path 无效，可在此覆盖新的绝对/相对路径

# === 从缓存三件套恢复数据视图 ===
def load_cached_datasets(cache_dir="./data_cache", meta_path=None, bin_path_override=None, map_path_override=None):
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

    # 2) 解析 bin_path（允许覆盖；支持相对路径）
    bin_path = bin_path_override or os.path.basename(meta["bin_path"])
    if not os.path.isabs(bin_path):
        base_dir = os.path.dirname(os.path.abspath(meta_path))
        bin_path = os.path.normpath(os.path.join(base_dir, bin_path))

    # 3) 定位 char_map.pkl（优先外部覆盖；其次用 txt_name 的 basename；最后兜底搜索）
    if map_path_override:
        map_path = map_path_override
    else:
        # 用原始 txt 文件名作为基名（与 TextDataProcessor 的保存逻辑一致）
        txt_basename = os.path.basename(meta.get("txt_name", ""))
        if not txt_basename:
            # 如果 meta 里没有 txt_name，就退化成通配搜索
            candidates = glob.glob(os.path.join(os.path.dirname(meta_path), "*.char_map.pkl"))
            if not candidates:
                raise FileNotFoundError("未找到 *.char_map.pkl（且 meta.json 中缺少 txt_name 字段）。")
            map_path = candidates[0]
        else:
            map_path = os.path.join(os.path.dirname(meta_path), f"{txt_basename}.char_map.pkl")
            if not os.path.exists(map_path):
                # 兜底：通配搜索
                candidates = glob.glob(os.path.join(os.path.dirname(meta_path), "*.char_map.pkl"))
                if not candidates:
                    raise FileNotFoundError(f"未找到映射文件：{map_path}")
                map_path = candidates[0]

    with open(map_path, "rb") as f:
        maps = pickle.load(f)
    char_to_idx = maps["char_to_idx"]
    idx_to_char = maps["idx_to_char"]
    vocab_size  = maps["vocab_size"]

    # 4) 尺寸校验（可选）
    dtype_bytes = 2 if meta["dtype"] == "uint16" else 4
    expect = int(meta["n_total"]) * dtype_bytes
    actual = os.path.getsize(bin_path)
    if expect != actual:
        raise ValueError(f".tokens.bin 尺寸不匹配：expect={expect}, actual={actual}, bin_path={bin_path}")

    # 5) 构建 memmap 视图
    n_total   = int(meta["n_total"])
    split_idx = int(meta["split_idx"])
    dtype     = meta["dtype"]

    train_ds = MemmapTokenDataset(bin_path, 0,            split_idx,           dtype=dtype)
    val_ds   = MemmapTokenDataset(bin_path, split_idx,    n_total - split_idx, dtype=dtype)

    return train_ds, val_ds, vocab_size, char_to_idx, idx_to_char, meta

# === 基于 memmap 视图的 batch 生成 ===
def make_get_batch(train_ds, val_ds, device, block_size):
    # 说明：MemmapTokenDataset 支持切片返回 torch.long，因此这里直接堆叠即可
    def get_batch(split, batch_size, block_size=block_size):
        ds = train_ds if split == "train" else val_ds
        # 预留一个位给 y 的右移
        max_start = len(ds) - block_size - 1
        if max_start <= 0:
            raise ValueError("数据过短，无法切出一个 block_size+1 的序列")
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([ds[i:i+block_size]       for i in ix])         # [B, T]
        y = torch.stack([ds[i+1:i+block_size+1]   for i in ix])         # [B, T]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return get_batch

@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters=200, device="cuda"):
    was_training = model.training
    model.eval()
    losses = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            acc = 0.0
            for _ in range(eval_iters):
                x, y = get_batch_fn(split, batch_size, block_size)
                logits, loss = model(x, y)
                acc += loss.item()
            losses[split] = acc / eval_iters
    if was_training:
        model.train()
    return losses


# === 训练主函数 ===
def train():
    # 1) 从缓存恢复数据（不需要 DataSaver）
    train_ds, val_ds, vocab_size, char_to_idx, idx_to_char, meta = load_cached_datasets()

    # 2) 准备模型
    print("【模型初始化】开始")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel( vocab_size, block_size, Embedding_dim, num_heads, numOfLayers, dropout).to(device)
    print(sum(p.numel() for p in model.parameters()), "参数总数")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 3) 批次函数
    get_batch_fn = make_get_batch(train_ds, val_ds, device, block_size)

    for i in range(iters):
        print(f"【模型训练迭代: {i+1}/{iters}】")
        if i % eval_interval == 0 or i == iters - 1: 
            losses = estimate_loss(model, get_batch_fn, eval_iters=eval_iters, device=device)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch_fn('train', batch_size, block_size)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model_weights.pth')

# === 入口 ===
if __name__ == "__main__":
    train()
