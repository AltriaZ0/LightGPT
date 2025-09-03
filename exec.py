from model_config import *
from model import LanguageModel
from data_save import DataSaver
import torch
import torch.nn.functional as F
import random

# 推理专用处理器
class InferenceProcessor:
    def __init__(self, state):
        self.char_to_idx = state['char_to_idx']
        self.idx_to_char = state['idx_to_char']
        self.vocab_size = state['vocab_size']
    
    def decode_text(self, indices):
        """将 token 索引列表转为字符串，忽略无效索引"""
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)  # 用 ? 代替未知索引
    
    def encode_text(self, text):
        """将字符串转为 token 索引列表，未知字符跳过或用默认值"""
        return [self.char_to_idx.get(c, 0) for c in text]  # 0 可视为 <unk> 或 padding


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


def output():
    # 1. 加载模型
    model = LanguageModel().cuda()
    model.eval()  # 重要：切换到评估模式

    # 2. 加载处理器状态
    persister = DataSaver()
    processor_state = persister.load_processor_state("data_cache/processor_state.pkl")
    processor = InferenceProcessor(processor_state)

    # 3. 加载数据（用于随机上下文和真实对比）
    data_tensors = persister.load_data_tensors("data_cache/data_tensors.pt")
    val_data = data_tensors['val_data']

    # 4. 加载权重
    model.load_state_dict(torch.load('model_weights.pth', map_location='cuda'))
    print("模型加载完成")

    # 5. 用户选择模式
    print("随机续写还是输入上文？(r: 随机续写, i: 输入上文)")
    choice = input().strip().lower()

    if choice == 'i':
        print("请输入上文:")
        input_text = input().strip()
        if not input_text:
            print("输入为空，使用默认起始符")
            input_text = "\n"
        input_tokens = processor.encode_text(input_text)
        if not input_tokens:
            print("输入无法编码，使用换行符作为起始")
            input_tokens = [processor.char_to_idx.get('\n', 0)]
        context = torch.tensor([input_tokens], dtype=torch.long).cuda()
    else:
        # 随机选择一个起始位置
        start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)
        context = val_data[start_idx:start_idx + block_size].unsqueeze(0).cuda()  # (1, block_size)

    # 6. 打印上文
    print("【模型输出】上文:", processor.decode_text(context[0].tolist()))

    # 7. 流式生成
    generated_full = generate_streaming(
        model=model,
        processor=processor,
        context=context,
        max_new_tokens=max_new_tokens,
        temperature=0.8,   # 控制多样性
        top_k=50           # 提高生成质量
    )

    # 8. 对比真实后续（仅在随机模式下有意义）
    if choice == 'r':
        real_next = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens]
        print("【真实文本】实际后续: \n" + processor.decode_text(real_next.tolist()))


# 运行
if __name__ == "__main__":
    output()