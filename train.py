from model import *
from data import *
from data_save import DataSaver

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
    
    for i in range(iters): 
        print(f"【模型训练迭代: {i+1}/{iters}】")
        if i % eval_interval == 0 or iter == iters - 1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch('train', batch_size, block_size)
        logits, loss = model(x, y) # 前馈运算
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # 反向传播，计算新的梯度
        optimizer.step()

    # save the model
    torch.save(model.state_dict(), 'model_weights.pth')

train()