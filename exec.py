from model_config import *
from model import LanguageModel
from data_save import DataSaver

# 创建文本处理器实例（仅用于解码）
class InferenceProcessor:
    def __init__(self, state):
        self.char_to_idx = state['char_to_idx']
        self.idx_to_char = state['idx_to_char']
        self.vocab_size = state['vocab_size']
    
    def decode_text(self, indices):
        return ''.join(self.idx_to_char[i] for i in indices if i in self.idx_to_char)

# def output():
#     model = LanguageModel().cuda() # 实例化一个模型

#     persister = DataSaver()
#     # 加载处理器状态
#     processor_state = persister.load_processor_state("data_cache/processor_state.pkl")
#     processor = InferenceProcessor(processor_state)

#     # 加载数据张量
#     data_tensors = persister.load_data_tensors("data_cache/data_tensors.pt")
#     val_data = data_tensors['val_data']
    
#     model.load_state_dict(torch.load('model_weights.pth')) # 加载模型参数
#     model.eval() # 切换到评估模式
#     start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)# 随机选择起始位置

#     # 创建上下文
#     context = torch.zeros(1, block_size, dtype=torch.long).cuda()
#     context[0, :] = val_data[start_idx:start_idx + block_size].cuda()
#     # 使用处理器解码上下文
#     print("【模型输出】上文:", processor.decode_text(context[0].tolist()))

#     # 生成文本
#     next_tokens = model.generate(context, max_new_tokens)

#     # 打印生成的文本
#     print("【模型输出】生成的文本: \n", processor.decode_text(next_tokens[0].tolist()))


#     # 获取实际的下一个token
#     real_next_tokens = torch.zeros(1, max_new_tokens, dtype=torch.long).cuda()
#     real_next_tokens[0, :] = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens].cuda()
    

#     # 使用处理器解码实际的下一个token
#     print("【模型输出】实际的下一个token: \n", processor.decode_text(real_next_tokens[0].tolist()))

def output():
    model = LanguageModel().cuda()  # 实例化模型
    persister = DataSaver()

    # 加载处理器状态
    processor_state = persister.load_processor_state("data_cache/processor_state.pkl")
    processor = InferenceProcessor(processor_state)

    # 加载数据张量
    data_tensors = persister.load_data_tensors("data_cache/data_tensors.pt")
    val_data = data_tensors['val_data']

    # 加载模型权重
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # 切换到评估模式

    # 随机选择起始位置
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)
    
    # 创建初始上下文
    context = torch.zeros(1, block_size, dtype=torch.long).cuda()
    context[0, :] = val_data[start_idx:start_idx + block_size].cuda()

    # 解码并打印上下文
    print("【模型输出】上文:", processor.decode_text(context[0].tolist()))
    print("【模型输出】生成的文本: \n", end="", flush=True)

    # 流式生成并输出
    generated_tokens = context.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 只取最后 block_size 个 token 作为输入
            input_tokens = generated_tokens[:, -block_size:]
            logits, _ = model(input_tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 添加到已生成序列
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # 解码最新 token 并输出
            char = processor.decode_text(next_token[0].tolist())
            print(char, end="", flush=True)  # 实时输出，不换行

    print()  # 换行结束

    # 打印真实的后续文本
    real_next_tokens = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens]
    print("【模型输出】实际的下一个token: \n", processor.decode_text(real_next_tokens.tolist()))

output()