import os
import io
import json
import pickle
import codecs
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from model_train_config import txt_name
from torch.autograd import Variable
import os
import pickle
from nvtx import nvtx_range

class MemmapTokenDataset(Dataset):
    """
    基于 numpy.memmap 的只读 1D token 数据集。
    支持随机索引和 __len__，几乎不占额外内存。
    """
    def __init__(self, bin_path, offset, length, dtype="uint32"):
        self.bin_path = bin_path
        self.offset = int(offset)
        self.length = int(length)
        self.dtype = np.dtype(dtype)
        # 只映射索引视图，不把数据读到内存
        self._mm = np.memmap(self.bin_path, mode="r", dtype=self.dtype)
        if self.offset + self.length > self._mm.shape[0]:
            raise ValueError("Dataset slice exceeds memmap length")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sl = slice(self.offset + (idx.start or 0),
                       self.offset + (idx.stop or self.length),
                       idx.step)
            arr = self._mm[sl]
            # 训练时通常需要 torch.long
            return torch.from_numpy(np.array(arr, copy=False)).long()
        else:
            i = self.offset + int(idx)
            val = int(self._mm[i])
            return torch.tensor(val, dtype=torch.long)

class TextDataProcessor:
    def __init__(self, txt_name, train_ratio=0.9, seed=42, cache_dir='./data_cache',
                 chunk_bytes=8*1024*1024,  # 4MiB 字节块，增量 UTF-8 解码
                 memmap_dtype='uint32'):
        """
        参数:
            txt_name: 文本文件路径
            train_ratio: 训练集比例（按字符数切分）
            seed: 随机种子（这里仅用于一致性）
            cache_dir: 缓存目录
            chunk_bytes: 读取块大小（字节），配合 UTF-8 增量解码安全处理
            memmap_dtype: memmap 的 dtype，按 vocab 大小选择 'uint16'/'uint32'
        """
        torch.manual_seed(seed)
        self.txt_name = txt_name
        self.train_ratio = float(train_ratio)
        self.cache_dir = cache_dir
        self.chunk_bytes = int(chunk_bytes)
        self.memmap_dtype = memmap_dtype  # 会在建表后自动下调到 uint16（若可行）

        self.text = None              # 不再常驻保存全文
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = 0

        # memmap 元数据
        self.meta = None              # dict，包含 n_total, split_idx, bin_path, dtype 等
        self._train_ds = None
        self._val_ds   = None

        os.makedirs(self.cache_dir, exist_ok=True)

        if self._load_from_cache():
            print("从缓存加载元数据和映射成功。")
        else:
            print("缓存缺失或不可用，开始流式处理文本……")
            self._process_and_build_cache()

        # 准备 Dataset 视图
        self._prepare_datasets()

    # ---------- 缓存文件名 ----------
    def _basename(self):
        return os.path.basename(self.txt_name)

    def _meta_path(self):
        return os.path.join(self.cache_dir, f"{self._basename()}.meta.json")

    def _map_path(self):
        return os.path.join(self.cache_dir, f"{self._basename()}.char_map.pkl")

    def _bin_path(self):
        return os.path.join(self.cache_dir, f"{self._basename()}.tokens.bin")

    # ---------- 缓存读写 ----------
    def _load_from_cache(self):
        try:
            if not (os.path.exists(self._meta_path()) and os.path.exists(self._map_path())):
                return False
            with open(self._meta_path(), 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
            with open(self._map_path(), 'rb') as f:
                maps = pickle.load(f)
            self.char_to_idx = maps['char_to_idx']
            self.idx_to_char = maps['idx_to_char']
            self.vocab_size  = maps['vocab_size']
            # 简单校验 bin
            if not os.path.exists(self.meta['bin_path']):
                print("发现元数据，但 .bin 不存在。将重新处理。")
                return False
            return True
        except Exception as e:
            print(f"缓存加载失败：{e}。将重新处理。")
            return False

    def  _save_cache(self):
        with open(self._meta_path(), 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(self._map_path(), 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"缓存已写入：\n - {self._meta_path()}\n - {self._map_path()}\n - {self._bin_path()}")

    # ---------- 核心：两遍扫描 ----------
    def _process_and_build_cache(self):
        # 第一遍：UTF-8 安全增量解码，收集字符集合与总长度
        decoder = codecs.getincrementaldecoder('utf-8')()
        uniq = set()
        n_total = 0

        with open(self.txt_name, 'rb') as fb:
            while True:
                print("n_total: ", n_total)
                chunk = fb.read(self.chunk_bytes)
                if not chunk:
                    break
                s = decoder.decode(chunk)
                if s:
                    uniq.update(s)
                    n_total += len(s)
            # flush 末尾不完整序列
            tail = decoder.decode(b'', final=True)
            if tail:
                uniq.update(tail)
                n_total += len(tail)
        print(f"总字符数: {n_total}, 词汇表大小: {len(uniq)}")
        # 建立映射
        chars = sorted(list(uniq))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # 选择 memmap dtype（若 vocab_size <= 65535 则用 uint16 节省空间）
        if self.vocab_size <= 65535:
            self.memmap_dtype = 'uint16'
        else:
            self.memmap_dtype = 'uint32'

        split_idx = int(n_total * self.train_ratio)
        bin_path = self._bin_path()

        # 第二遍：重新增量解码，把编码索引写入 memmap
        mm = np.memmap(bin_path, mode='w+', dtype=self.memmap_dtype, shape=(n_total,))
        decoder = codecs.getincrementaldecoder('utf-8')()
        write_pos = 0

        with open(self.txt_name, 'rb') as fb:
            while True:
                print("write_pos: ", write_pos)
                chunk = fb.read(self.chunk_bytes)
                if not chunk:
                    break
                s = decoder.decode(chunk)
                if not s:
                    continue
                # 批量编码
                # 注意：过滤不在映射内的字符（理论上不会发生，因为第一遍已覆盖）
                idx_arr = np.fromiter((self.char_to_idx.get(c, 0) for c in s),
                                      dtype=self.memmap_dtype, count=len(s))
                end = write_pos + len(idx_arr)
                mm[write_pos:end] = idx_arr
                write_pos = end

            tail = decoder.decode(b'', final=True)
            if tail:
                idx_arr = np.fromiter((self.char_to_idx.get(c, 0) for c in tail),
                                      dtype=self.memmap_dtype, count=len(tail))
                end = write_pos + len(idx_arr)
                mm[write_pos:end] = idx_arr
                write_pos = end

        mm.flush()
        del mm  # 关闭映射

        self.meta = {
            'txt_name': self.txt_name,
            'bin_path': bin_path,
            'n_total': n_total,
            'split_idx': split_idx,
            'train_ratio': self.train_ratio,
            'dtype': self.memmap_dtype,
            'seed': 42
        }
        self._save_cache()

    # ---------- Dataset 视图 ----------
    def _prepare_datasets(self):
        n_total = self.meta['n_total']
        split_idx = self.meta['split_idx']
        bin_path = self.meta['bin_path']
        dtype    = self.meta['dtype']

        self._train_ds = MemmapTokenDataset(bin_path, 0, split_idx, dtype=dtype)
        self._val_ds   = MemmapTokenDataset(bin_path, split_idx, n_total - split_idx, dtype=dtype)

    # ---------- 兼容原 API（做了更安全的调整） ----------
    def _warn_tensor_materialize(self):
        print("提示：为了避免内存爆炸，不再返回整段大 Tensor。请改用 get_train_dataset()/get_val_dataset() "
              "并在 DataLoader 中按需取用。确需拿到一小段可用 torch.LongTensor，请使用切片："
              "dataset[beg:end]。")

    @nvtx_range()
    def get_train_data(self):
        self._warn_tensor_materialize()
        return self._train_ds

    @nvtx_range()
    def get_val_data(self):
        self._warn_tensor_materialize()
        return self._val_ds

    @nvtx_range()
    def get_train_dataset(self):
        return self._train_ds

    @nvtx_range()
    def get_val_dataset(self):
        return self._val_ds

    # ---------- 其它工具函数 ----------
    @nvtx_range()
    def encode_text(self, text):
        """将文本转换为索引列表（小段用；大段仍建议流式写入）"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    @nvtx_range()
    def decode_text(self, indices):
        return ''.join(self.idx_to_char[int(i)] for i in indices
                       if int(i) in self.idx_to_char)

    @nvtx_range()
    def get_vocab_size(self):
        return self.vocab_size

    @nvtx_range()
    def save_mappings(self, dir_path='.'):
        os.makedirs(dir_path, exist_ok=True)
        char_path = os.path.join(dir_path, 'char_mappings.txt')
        with open(char_path, 'w', encoding='utf-8') as f:
            for char, idx in self.char_to_idx.items():
                display_char = char
                if char == '\n':
                    display_char = '\\n'
                elif char == '\t':
                    display_char = '\\t'
                elif char == '\r':
                    display_char = '\\r'
                f.write(f"字符: {display_char} -> 索引: {idx}\n")

    def clear_cache(self):
        for p in [self._meta_path(), self._map_path(), self._bin_path()]:
            if os.path.exists(p):
                os.remove(p)
                print(f"已删除缓存: {p}")

    def print_summary(self):
        n_total = self.meta['n_total']
        split_idx = self.meta['split_idx']
        print(f"【文本数据处理器】文件: {self.txt_name}")
        print(f"【总字符数】: {n_total}")
        print(f"【词汇表大小】: {self.vocab_size}（dtype={self.meta['dtype']}）")
        print(f"【训练集】长度: {split_idx}")
        print(f"【验证集】长度: {n_total - split_idx}")

        # 小样本编解码测试（不拉整段内存）
        sample = self._train_ds[:64]  # 取前 64 个 token
        decoded = self.decode_text(sample.tolist())
        print(f"【编解码测试】前 64 token -> 文本片段: '{decoded}'")


# 使用示例
if __name__ == '__main__':

    processor = TextDataProcessor(txt_name)

    processor.print_summary()


