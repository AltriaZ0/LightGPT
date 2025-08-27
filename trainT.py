from model import *
from data import *
from data_save import DataSaver
import torch.cuda.nvtx as nvtx  # 导入NVTX模块

# 创建数据持久化对象
saver = DataSaver()
# 保存处理器状态 (包含字符映射等元数据)
processor_path = saver.save_processor_state(processor)
# 保存数据张量 (训练集和验证集)
tensors_path = saver.save_data_tensors(processor.get_train_data(), processor.get_val_data())

# 训练主函数
def train():
    print("【模型初始化】开始")
    model = LanguageModel().cuda()
    print(sum(p.numel() for p in model.parameters()), "参数总数")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 启动CUDA分析器
    torch.cuda.cudart().cudaProfilerStart()
    
    for i in range(iters): 
        print(f"【模型训练迭代: {i+1}/{iters}】")
        if i % eval_interval == 0 or i == iters - 1:  # 修正了变量名错误(iter->i)
            nvtx.range_push("Evaluation")
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            nvtx.range_pop()  # Evaluation结束

        # 标记训练迭代开始
        nvtx.range_push(f"Iteration {i}")
        
        # 获取批次数据
        nvtx.range_push("Data Loading")
        x, y = get_batch('train', batch_size, block_size)
        nvtx.range_pop()  # Data Loading结束
        
        # 前向传播
        nvtx.range_push("Forward Pass")
        logits, loss = model(x, y)  # 前馈运算
        nvtx.range_pop()  # Forward Pass结束
        
        # 反向传播
        nvtx.range_push("Backward Pass")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 反向传播，计算新的梯度
        nvtx.range_pop()  # Backward Pass结束
        
        # 参数更新
        nvtx.range_push("Optimizer Step")
        optimizer.step()
        nvtx.range_pop()  # Optimizer Step结束
        
        nvtx.range_pop()  # Iteration结束
        
        # 只分析前几次迭代以避免生成过大的报告文件
        if i == 4:  # 分析5次迭代(0-4)
            torch.cuda.cudart().cudaProfilerStop()
            break

    # 保存模型
    torch.save(model.state_dict(), 'model_weights.pth')

train()