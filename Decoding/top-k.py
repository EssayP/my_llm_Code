import torch
import torch.nn.functional as F

def top_k_sampling(logits:torch.Tensor,top_k:int=50,temperature:float=1.0):
    """
        Top-K Sampling 函数

        参数:
            logits: 模型输出的原始 logits，形状 (batch_size, vocab_size) 或 (vocab_size,)
            top_k: 只保留概率最高的 top_k 个 token（推荐 20~50）
            temperature: 温度参数，控制随机性（>1 更随机，<1 更确定）

        返回:
            next_token_id: 采样得到的下一个 token id，形状 (batch_size, 1) 或 scalar
        """
    if logits.dim() == 1:
        logits = logits.unsequeeze(0)

    batch_size,vocab_size = logits.size()

    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k_logits, _ = torch.topk(logits, min(top_k,vocab_size),dim=1)
        kth_values = top_k_logits[:,-1].unsqueeze(-1)
        indices_to_remove = logits < kth_values
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits,dim=-1)
    next_token = torch.multinomial(probs,num_samples=1)

    return next_token.squeeze(-1)

def generate_with_topk(model,tokenizer,prompt,max_new_tokens=100,top_k=50,temperature=0.8):
    model.eval()
    input_ids = tokenizer.encode(prompt,return_tensors="pt").to(model.device)

    generated = input_ids.clone()
    print("生成中...",end="")
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:,-1,:]
            next_token_id = top_k_sampling(next_token_logits,top_k=top_k,temperature=temperature)

        generated = torch.cat([generated,next_token_id.unsqueeze(-1)],dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0],skip_special_tokens=True)
    print("\n生成完成")
    return text

