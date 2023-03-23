import os
import torch
import wandb
from tqdm import tqdm, trange
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy

class ModelTrainer():
    
    def __init__(self, model, loss_fn, optimizer, device, save_dir, 
                 train_dataloader, valid_dataloader=None, 
                 epochs:int=1, base_score:float=0.45, num_classes:int=7):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.num_classes = num_classes
        self.base_score = base_score
        
    def train(self):
        self.model.to(self.device)
        for epoch in trange(self.epochs):
            self._fit()
            val_loss = self._validation()
            if self.base_score >=val_loss:
                self.base_score=val_loss
                self.model
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                torch.save(self.model.state_dict(), f'{self.save_dir}/epoch:{epoch}_model.pt')
                print(f'{epoch}epoch Model saved..!')
        
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
    
    def _fit(self):
        # Train
        self.model.train()
        pbar = tqdm(self.train_dataloader)
        epoch_loss=0
        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            cls_label = batch['label'].to(self.device)
            audio_tensor = batch['audio_emb'].to(self.device)
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            
            output = self.model(input_ids, 
                            attention_mask,
                            token_type_ids,
                            audio_tensor 
                            )['class_logit']
            
            loss = self.loss_fn(output, cls_label)
            loss.backward()
            self.optimizer.step()
            
            step_loss = loss.detach().cpu().item()
            wandb.log({'loss':step_loss})
            
            pbar.set_postfix({'loss': step_loss, 
                        "lr": self.optimizer.param_groups[0]["lr"]})
        pbar.close()
        
    def _validation(self):
        # Validation
        self.model.eval()
        pbar=tqdm(self.valid_dataloader)
        val_epoch_loss=0
        output_list=[]
        label_list=[]
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                cls_label = batch['label'].to(self.device)
                audio_tensor = batch['audio_emb'].to(self.device)
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                
                output = self.model(input_ids, 
                                attention_mask,
                                token_type_ids,
                                audio_tensor 
                                )['class_logit']
                
                valid_step_loss = self.loss_fn(output, cls_label)
                val_epoch_loss+=valid_step_loss.detach().cpu().item()
                
                output_list.append(output.detach().cpu())
                label_list.append(cls_label.detach().cpu())
        
        m_f1 = self._macro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)
        val_epoch_loss/=(step+1)

        wandb.log({'val_epoch_loss':val_epoch_loss,
            'val_macro_f1_score':m_f1,
            'val_weighted_f1_score':w_f1,
            'val_accuracy':acc})
        print(f"val_loss:{val_epoch_loss} macro_f1_score:{m_f1}")
        return val_epoch_loss

    def _macro_f1_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        m_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=self.num_classes, 
                                         average="macro").detach().cpu().item()
        return m_f1_score
    
    def _weighted_f1_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        w_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=self.num_classes, 
                                         average="weighted").detach().cpu().item()
        return w_f1_score
    
    def _accuracy_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        acc_score = multiclass_accuracy(logits, labels, 
                                         num_classes=self.num_classes).detach().cpu().item()
        return acc_score
    