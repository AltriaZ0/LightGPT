import torch
import pickle
import os
from data import TextDataProcessor

class DataSaver:
    def __init__(self, data_dir="data_cache"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_processor_state(self, processor, filename="processor_state.pkl"):
        """保存文本处理器状态"""
        state = {
            'char_to_idx': processor.char_to_idx,
            'idx_to_char': processor.idx_to_char,
            'vocab_size': processor.vocab_size,
            'train_data': processor.train_data,
            'val_data': processor.val_data,
            'text': processor.text  # 可选保存原始文本
        }
        path = os.path.join(self.data_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        return path
    
    def save_data_tensors(self, train_data, val_data, filename="data_tensors.pt"):
        """直接保存数据张量"""
        path = os.path.join(self.data_dir, filename)
        torch.save({
            'train_data': train_data,
            'val_data': val_data
        }, path)
        return path
    
    @staticmethod
    def load_processor_state(path):
        """加载文本处理器状态"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        return state
    
    @staticmethod
    def load_data_tensors(path):
        """加载数据张量"""
        return torch.load(path)