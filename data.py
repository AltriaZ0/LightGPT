import torch
from torch.nn import functional as F
from model_config import *
from torch.autograd import Variable
import os
import pickle
from nvtx import nvtx_range

class TextDataProcessor:
    def __init__(self, txt_name, train_ratio=0.9, seed=42, cache_dir='./data_cache'):
        """
        初始化文本数据处理器
        
        参数:
            txt_name: 文本文件路径
            train_ratio: 训练集比例
            seed: 随机种子
            cache_dir: 缓存目录
        """
        torch.manual_seed(seed)
        self.txt_name = txt_name
        self.train_ratio = train_ratio
        self.cache_dir = cache_dir
        self.text = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.train_data = None
        self.val_data = None
        self.vocab_size = 0
        
        # 检查缓存是否存在
        if self._check_cache():
            self._load_from_cache()
        else:
            self._load_data()
            self._save_to_cache()
    
    def _get_cache_filename(self):
        """生成缓存文件名"""
        # 使用文件名和训练比例生成唯一的缓存标识
        base_name = os.path.basename(self.txt_name)
        cache_name = f"{base_name}_ratio{self.train_ratio}_seed{42}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def _check_cache(self):
        """检查缓存文件是否存在"""
        cache_file = self._get_cache_filename()
        return os.path.exists(cache_file)
    
    def _save_to_cache(self):
        """保存处理结果到缓存"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = self._get_cache_filename()
        
        cache_data = {
            'text': self.text,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'train_data': self.train_data,
            'val_data': self.val_data,
            'vocab_size': self.vocab_size
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"数据已缓存到: {cache_file}")
    
    def _load_from_cache(self):
        """从缓存加载处理结果"""
        cache_file = self._get_cache_filename()
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.text = cache_data['text']
            self.char_to_idx = cache_data['char_to_idx']
            self.idx_to_char = cache_data['idx_to_char']
            self.train_data = cache_data['train_data']
            self.val_data = cache_data['val_data']
            self.vocab_size = cache_data['vocab_size']
            
            print(f"从缓存加载数据: {cache_file}")
            return True
            
        except (pickle.PickleError, FileNotFoundError, KeyError) as e:
            print(f"缓存加载失败: {e}，将重新处理数据")
            return False
    
    def _load_data(self):
        """加载并处理数据"""
        print("正在处理文本数据...")
        # 读取轻小说文本
        with open(self.txt_name, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # 创建字符集
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        
        # 创建映射
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # 编码文本
        text_indices = [self.char_to_idx[c] for c in self.text]
        data = torch.tensor(text_indices, dtype=torch.long)
        
        # 划分训练集和验证集
        n = len(text_indices)
        split_idx = int(n * self.train_ratio)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
    
    def clear_cache(self):
        """清除当前实例的缓存"""
        cache_file = self._get_cache_filename()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"已清除缓存: {cache_file}")
        else:
            print("缓存文件不存在")
    @nvtx_range()
    def encode_text(self, text):
        """将文本转换为索引列表"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    @nvtx_range()
    def decode_text(self, indices):
        """将索引列表转换为文本"""
        return ''.join(self.idx_to_char[i] for i in indices if i in self.idx_to_char)
    @nvtx_range()
    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.vocab_size
    @nvtx_range()
    def get_train_data(self):
        """获取训练数据"""
        return self.train_data
    @nvtx_range()
    def get_val_data(self):
        """获取验证数据"""
        return self.val_data
    @nvtx_range()
    def save_mappings(self, dir_path='.'):
        """保存字符映射到文件"""
        os.makedirs(dir_path, exist_ok=True)
        char_path = os.path.join(dir_path, 'char_mappings.txt')
        
        with open(char_path, 'w', encoding='utf-8') as f:
            for char, idx in self.char_to_idx.items():
                # 处理特殊字符的显示
                display_char = char
                if char == '\n':
                    display_char = '\\n'
                elif char == '\t':
                    display_char = '\\t'
                elif char == '\r':
                    display_char = '\\r'
                f.write(f"字符: {display_char} -> 索引: {idx}\n")
    
    def print_summary(self):
        """打印数据摘要"""
        print(f"【文本数据处理器】文件: {self.txt_name}")
        print(f"【文本长度】: {len(self.text)}")
        print(f"【词汇表大小】: {self.vocab_size}")
        print(f"【训练集】长度: {len(self.train_data)}")
        print(f"【验证集】长度: {len(self.val_data)}")
        
        # 测试编码和解码
        test_str = "绫濑小姐，你好!"
        encoded = self.encode_text(test_str)
        decoded = self.decode_text(encoded)
        print(f"【编解码测试】原始: '{test_str}' -> 编码: {encoded} -> 解码: '{decoded}'")


# 使用示例 - 只有在直接运行data.py时才会执行
if __name__ == '__main__':
    # 创建处理器实例
    processor = TextDataProcessor(txt_name)
    
    # 打印摘要信息
    processor.print_summary()
    
    # 保存映射文件
    processor.save_mappings()
    
    # 测试：创建一个处理器实例
    processor2 = TextDataProcessor(txt_name)
    processor2.print_summary()
    
    # 清除缓存（可选）
    # processor.clear_cache()