import torch
import torch.nn.functional as F
class DPO:
    def __init__(self,beta):
        self.beta = beta

    # 计算 logπθ(y∣x)
    def get_logps(self,logits,labels,mask):
        """

        :param logits: [batch_size,seq_len,vocab_size]
        :param labels: [batch_size,seq_len]
        :param mask: [batch_size,seq_len)
        :return:
        """
        log_probs = F.log_softmax(logits,dim=-1)
        # 把每个位置真实 token 对应的 log probability 提取出来
        # 在 log_probs 的最后一个维度（vocab_size）上，
        # 根据 labels 提供的索引，取出对应的 log prob
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1) #[batch_size,seq_len]
        seq_logps = (token_log_probs * mask).sum(dim=-1)
        return seq_logps

    def dpo_loss(self,policy_chosen_logps,policy_reject_logps,ref_chosen_logps,ref_reject_logps):
        chosen_r = policy_chosen_logps - ref_chosen_logps
        reject_r = policy_reject_logps - ref_reject_logps
        loss = -F.logsigmoid(self.beta * (chosen_r - reject_r))
        return loss.mean()

    def forward(self,
                model,
                ref_model,
                input_ids_chosen,
                labels_chosen,
                mask_chosen,
                input_ids_reject,
                labels_reject,
                mask_reject):
        # policy model
        logits_chosen = model(input_ids_chosen).logits
        logits_reject = model(input_ids_reject).logits

        policy_chosen_logps = self.get_logps(logits_chosen,labels_chosen,mask_chosen)
        policy_reject_logps = self.get_logps(logits_reject,labels_reject,mask_reject)

        # ref model
        with torch.no_grad():
            ref_logits_chosen = ref_model(input_ids_chosen).logits
            ref_logits_reject = ref_model(input_ids_reject).logits
            ref_chosen_logps = self.get_logps(ref_logits_chosen,labels_chosen,mask_chosen)
            ref_logits_reject = self.get_logps(ref_logits_reject,labels_reject,mask_reject)

        loss = self.dpo_loss(policy_chosen_logps,
                             policy_reject_logps,
                             ref_chosen_logps,
                             ref_logits_reject)

        return loss

