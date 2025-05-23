from transformers import Trainer
import torch.nn.functional as F
import torch
from trl import GKDTrainer

class CustomGKDTrainer(GKDTrainer):
    def __init__(self, ce_weight: float = 0, gkd_weight: float = 1.0, *args, **kwargs):
        """
        Args:
          ce_weight: multiplier for cross‐entropy loss
          gkd_weight: multiplier for GKD loss
        """
        super().__init__(*args, **kwargs)
        self.ce_weight = ce_weight
        self.gkd_weight = gkd_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1) Run the student forward pass
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )
        ce_loss = outputs_student.loss

        # 2) Compute standard CE loss
        #    HuggingFace will have applied shift internally if labels passed to Trainer,
        #    but to be explicit we can do:
        # labels = inputs["labels"]
        # # Flatten (batch*seq, vocab) & (batch*seq,)
        # ce_loss = F.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     labels.view(-1),
        #     ignore_index=-100,
        # )

        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # compute loss
        gkd_loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )

        # 4) Combine
        total_loss = self.ce_weight * ce_loss + self.gkd_weight * gkd_loss

        # 5) Log both losses (Trainer.log will push to WandB automatically)
        logs = {
            "loss_ce": ce_loss.detach().cpu().item(),
            #"loss_gkd": gkd_loss.detach().cpu().item(),
            #"loss": total_loss.detach().cpu().item(),
        }
        # Trainer.log is only available after init; we can use self.log here:
        self.log(logs)


        empty_cache()

        # Return loss
        return (total_loss, outputs_student) if return_outputs else total_loss
