import os
import torch
from transformers import get_linear_schedule_with_warmup
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from tqdm import tqdm


def test(model, tese_dataloader, device, num_classes):
    model.eval()
    output_list=[]
    label_list=[]
    with torch.no_grad():
        for batch in tese_dataloader:
            cls_label = batch['label'].to(device)
            audio_tensor = batch['audio_emb'].to(device)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            
            output = model(input_ids, 
                            attention_mask,
                            token_type_ids,
                            audio_tensor 
                            )
            
            output_list.append(output.detach().cpu())
            label_list.append(cls_label.detach().cpu())
        
        logits = torch.cat(output_list)
        labels = torch.cat(label_list)
        
        m_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=num_classes, 
                                         average="macro").detach().cpu().item()
        w_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=num_classes, 
                                         average="weighted").detach().cpu().item()
        acc_score = multiclass_accuracy(logits, labels, 
                                         num_classes=num_classes, 
                                         average="weighted").detach().cpu().item()
        
        print(f'Macro F1 Score: {m_f1_score}, Weighted F1 Score: {w_f1_score}, Accuracy: {acc_score}')