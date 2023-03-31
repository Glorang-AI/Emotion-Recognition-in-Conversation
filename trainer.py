import os
import torch
import wandb

from tqdm import tqdm, trange
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from transformers import AutoTokenizer
from utils.contrastive import contrastive_set

class ModelTrainer():
    
    def __init__(self, args, 
                 model, loss_fn, optimizer,  
                 train_dataloader, valid_dataloader=None, test_dataloader=None,
                 scheduler = None, contrastive_loss_fn = None,
                 label_dict=None, verbalizer_value=None,
                 base_score:float=0.45):
        
        self.args = args # args: device, loss_fn, optimizer, save_dir
        
        self.model = model
        self.loss_fn = loss_fn
        self.contrastive_loss_fn = contrastive_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        
        self.base_score = base_score

        if verbalizer_value!=None and args.pet:
            self.verbalizer_value = list(verbalizer_value.values())
            
        else:
            self.verbalizer_value = None

        self.label_dict = label_dict
        self.neutral_label = label_dict['neutral']
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.lm_path)
        
        
    def train(self):
        self.model.to(self.args.device)
        for epoch in trange(self.args.epochs):

            if self.args.model == "speech_only":
                self._train_speech()
                if self.args.val_ratio:
                    val_loss = self._validation_speech() # Validation

                self._test_speech() # Test

            else:
                self._train() # Train

                if self.args.val_ratio:
                    val_loss = self._validation() # Validation

                self._test() # Test

            # if self.base_score >= val_loss:

            #     self.base_score=val_loss
            #     if not os.path.exists(self.args.save_path):
            #         os.makedirs(self.args.save_path)

            #     torch.save(self.model.state_dict(), f'{self.args.save_path}/epoch:{epoch}_{self.args.model}model_shceduler-{self.args.scheduler}_{self.args.contrastive}.pt')
            #     print(f'{epoch}epoch Model saved..!')
        
        torch.cuda.empty_cache()

        if self.args.val_ratio:
            del self.model, self.train_dataloader, self.valid_dataloader
        else:
            del self.model, self.train_dataloader
    
    def _train(self):
        # Train
        self.model.train()
        
        train_epoch_loss = 0
        output_list = []
        label_list = []

        pbar = tqdm(self.train_dataloader)
        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            label = batch['label'].to(self.args.device)
            audio_tensor = batch['audio_emb'].to(self.args.device)
            
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            token_type_ids = batch["token_type_ids"].to(self.args.device)
            
            output = self.model(
                input_ids, 
                attention_mask,
                token_type_ids,
                audio_tensor 
                )
            logit = output['class_logit']
            if self.args.pet: 
                # PET
                output = output['prediction_scores']
                _, mask_pos = torch.where(input_ids==self.tokenizer.mask_token_id)
                self.verbalizer_idx = self.tokenizer(self.verbalizer_value, 
                                                              add_special_tokens=False, 
                                                              return_tensors='pt').input_ids.squeeze().to(self.args.device)
               
                # Verbalizer Label 토큰에 대한 logit값
                logit = torch.stack([pred_score[mask_idx, :][self.verbalizer_idx] for pred_score, mask_idx in zip(output, mask_pos)])
                # print(output.shape, mask_pos.shape)
                loss = self.loss_fn(logit, label)
                
            else:
                # ORG
                logit = output['class_logit']
                loss = self.loss_fn(logit, label)

                if self.args.contrastive:
                    # Contrastive
                    pooled_output = output['pooled_output']
                    contrastive_value, contrastive_label = contrastive_set(pooled_output, label)
                    contrastive_loss = self.contrastive_loss_fn(contrastive_value, contrastive_label)
                    loss += 0.1 * contrastive_loss

            loss.backward()

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            step_loss = loss.detach().cpu().item()
            train_epoch_loss += step_loss

            wandb.log({'loss':step_loss})
            
            output_list.append(logit.detach().cpu())
            label_list.append(label.detach().cpu())

            pbar.set_postfix({'loss': step_loss, 
                        "lr": self.optimizer.param_groups[0]["lr"]})
        
        train_epoch_loss /= (step+1)
        m_f1 = self._macro_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)
        
        wandb.log({'train_macro_f1_score':m_f1,
                   'train_weighted_f1_score':w_f1,
                   'train_micro_f1_score':mic_f1,
                   'train_accuracy':acc})

        print(f"Train Loss: {train_epoch_loss: .4f} \nTrain Acc: {acc :.4f} \
            \nTrain Macro-F1: {m_f1:.4f} \nTrain Weighted-F1: {w_f1:.4f} \nTrain Micro-F1: {mic_f1:.4f}")

        pbar.close()

    def _train_speech(self):
        # Train
        self.model.train()
        
        train_epoch_loss = 0
        output_list = []
        label_list = []

        pbar = tqdm(self.train_dataloader)
        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            label = batch['label'].to(self.args.device)
            audio_tensor = batch['audio_emb'].to(self.args.device)
            
            output = self.model(audio_tensor)

            logit = output['class_logit']
            loss = self.loss_fn(logit, label)

            if self.args.contrastive:
                # Contrastive
                pooled_output = output['pooled_output']
                contrastive_value, contrastive_label = contrastive_set(pooled_output, label)
                contrastive_loss = self.contrastive_loss_fn(contrastive_value, contrastive_label)
                loss += 0.1 * contrastive_loss

            loss.backward()

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            step_loss = loss.detach().cpu().item()
            train_epoch_loss += step_loss

            wandb.log({'loss':step_loss})
            
            output_list.append(logit.detach().cpu())
            label_list.append(label.detach().cpu())

            pbar.set_postfix({'loss': step_loss, 
                        "lr": self.optimizer.param_groups[0]["lr"]})
        
        train_epoch_loss /= (step+1)
        m_f1 = self._macro_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)
        
        wandb.log({'train_macro_f1_score':m_f1,
                   'train_weighted_f1_score':w_f1,
                   'train_micro_f1_score':mic_f1,
                   'train_accuracy':acc})

        print(f"Train Loss: {train_epoch_loss: .4f} \nTrain Acc: {acc :.4f} \
            \nTrain Macro-F1: {m_f1:.4f} \nTrain Weighted-F1: {w_f1:.4f} \nTrain Micro-F1: {mic_f1:.4f}")

        pbar.close()
        
    def _validation(self):
            # Validation
        self.model.eval()

        val_epoch_loss=0
        output_list=[]
        label_list=[]
        with torch.no_grad():

            pbar=tqdm(self.valid_dataloader)
            for step, batch in enumerate(pbar):

                label = batch['label'].to(self.args.device)
                audio_tensor = batch['audio_emb'].to(self.args.device)
                
                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)
                token_type_ids = batch["token_type_ids"].to(self.args.device)
                
                output = self.model(
                    input_ids, 
                    attention_mask,
                    token_type_ids,
                    audio_tensor 
                    )
                
                if self.args.pet: 
                    # PET
                    output = output['prediction_scores']
                    _, mask_pos = torch.where(input_ids==self.tokenizer.mask_token_id)
                    self.verbalizer_idx = self.tokenizer(self.verbalizer_value, 
                                                                add_special_tokens=False, 
                                                                return_tensors='pt').input_ids.squeeze().to(self.args.device)
                    
                    # Verbalizer Label 토큰에 대한 logit값
                    logit = torch.stack([pred_score[mask_idx, :][self.verbalizer_idx] for pred_score, mask_idx in zip(output, mask_pos)])
                    valid_step_loss = self.loss_fn(logit, label)
                
                else:
                    logit = output['class_logit']
                    valid_step_loss = self.loss_fn(logit, label)
                    
                val_epoch_loss += valid_step_loss.detach().cpu().item()
                
                output_list.append(logit.detach().cpu())
                label_list.append(label.detach().cpu())
        
        m_f1 = self._macro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)

        val_epoch_loss /= (step+1)

        wandb.log({'val_epoch_loss':val_epoch_loss,
                   'val_macro_f1_score':m_f1,
                   'val_weighted_f1_score':w_f1,
                   'val_micro_f1_score':mic_f1,
                   'val_accuracy':acc})
        
        y_pred = torch.cat(output_list)
        y_test = torch.cat(label_list)
        
        print(f"Valid Loss: {val_epoch_loss: .4f} \nValid Acc: {acc :.4f} \
              \nValid Macro-F1: {m_f1:.4f} \nValid Weighted-F1: {w_f1:.4f} \nValid Micro-F1: {mic_f1:.4f}")
        
        return val_epoch_loss

    def _validation_speech(self):
        # Validation
        self.model.eval()

        val_epoch_loss=0
        output_list=[]
        label_list=[]
        with torch.no_grad():

            pbar=tqdm(self.valid_dataloader)
            for step, batch in enumerate(pbar):

                label = batch['label'].to(self.args.device)
                audio_tensor = batch['audio_emb'].to(self.args.device)
                
                output = self.model(audio_tensor)

                logit = output['class_logit']
                valid_step_loss = self.loss_fn(logit, label)
                    
                val_epoch_loss += valid_step_loss.detach().cpu().item()
                
                output_list.append(logit.detach().cpu())
                label_list.append(label.detach().cpu())
        
        m_f1 = self._macro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)

        val_epoch_loss /= (step+1)

        wandb.log({'val_epoch_loss':val_epoch_loss,
                   'val_macro_f1_score':m_f1,
                   'val_weighted_f1_score':w_f1,
                   'val_micro_f1_score':mic_f1,
                   'val_accuracy':acc})
        
        y_pred = torch.cat(output_list)
        y_test = torch.cat(label_list)
        
        print(f"Valid Loss: {val_epoch_loss: .4f} \nValid Acc: {acc :.4f} \
              \nValid Macro-F1: {m_f1:.4f} \nValid Weighted-F1: {w_f1:.4f} \nValid Micro-F1: {mic_f1:.4f}")
        
        return val_epoch_loss
    


    def _test(self):
        # Validation
        self.model.eval()

        output_list=[]
        label_list=[]
        with torch.no_grad():

            pbar=tqdm(self.test_dataloader)
            for step, batch in enumerate(pbar):

                label = batch['label'].to(self.args.device)
                audio_tensor = batch['audio_emb'].to(self.args.device)
                
                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)
                token_type_ids = batch["token_type_ids"].to(self.args.device)
                
                output = self.model(
                    input_ids, 
                    attention_mask,
                    token_type_ids,
                    audio_tensor 
                    )
                
                if self.args.pet: 
                    # PET
                    output = output['prediction_scores']
                    _, mask_pos = torch.where(input_ids==self.tokenizer.mask_token_id)
                    self.verbalizer_idx = self.tokenizer(self.verbalizer_value, 
                                                                add_special_tokens=False, 
                                                                return_tensors='pt').input_ids.squeeze().to(self.args.device)
                    
                    # Verbalizer Label 토큰에 대한 logit값
                    output = torch.stack([pred_score[mask_idx, :][self.verbalizer_idx] for pred_score, mask_idx in zip(output, mask_pos)])
                    
                else:
                    output = output['class_logit']
                
                output_list.append(output.detach().cpu())
                label_list.append(label.detach().cpu())

        m_f1 = self._macro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)

        wandb.log({'test_macro_f1_score':m_f1,
                   'test_weighted_f1_score':w_f1,
                   'test_micro_f1_score':mic_f1,
                   'test_accuracy':acc})
        
        labels = list(self.label_dict.keys())
        y_pred = torch.argmax(torch.cat(output_list), dim=1).tolist()
        y_test = torch.cat(label_list).tolist()
        wandb.log({'Confusion Matrix':wandb.plot.confusion_matrix(probs=None, y_true=y_test,
                                                                  preds = y_pred, class_names=labels)})
        
    def _test_speech(self):
        # Validation
        self.model.eval()

        output_list=[]
        label_list=[]
        with torch.no_grad():

            pbar=tqdm(self.test_dataloader)
            for step, batch in enumerate(pbar):

                label = batch['label'].to(self.args.device)
                audio_tensor = batch['audio_emb'].to(self.args.device)
                
                output = self.model(audio_tensor)

                output = output['class_logit']
                
                output_list.append(output.detach().cpu())
                label_list.append(label.detach().cpu())

        m_f1 = self._macro_f1_score(output_list, label_list)
        w_f1 = self._weighted_f1_score(output_list, label_list)
        mic_f1 = self._micro_f1_score(output_list, label_list)
        acc = self._accuracy_score(output_list, label_list)

        wandb.log({'test_macro_f1_score':m_f1,
                   'test_weighted_f1_score':w_f1,
                   'test_micro_f1_score':mic_f1,
                   'test_accuracy':acc})
        
        labels = list(self.label_dict.keys())
        y_pred = torch.argmax(torch.cat(output_list), dim=1).tolist()
        y_test = torch.cat(label_list).tolist()
        wandb.log({'Confusion Matrix':wandb.plot.confusion_matrix(probs=None, y_true=y_test,
                                                                  preds = y_pred, class_names=labels)})

    def _macro_f1_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        m_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=self.args.num_labels, 
                                         average="macro").detach().cpu().item()
        return m_f1_score
    
    def _weighted_f1_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        w_f1_score = multiclass_f1_score(logits, labels, 
                                         num_classes=self.args.num_labels, 
                                         average="weighted").detach().cpu().item()
        return w_f1_score
    
    def _micro_f1_score(self, logit_list, label_list):
        labels = torch.cat(label_list)
        label_pos =torch.where(labels!=self.neutral_label)
        
        labels = labels[label_pos]
        logits = torch.cat(logit_list)[label_pos]
        logits = torch.argmax(logits, dim=1)
        micro_f1_score = multiclass_f1_score(logits, labels, 
                                         average="micro").detach().cpu().item()
        return micro_f1_score
    
    
    def _accuracy_score(self, logit_list, label_list):
        logits = torch.cat(logit_list)
        labels = torch.cat(label_list)
        acc_score = multiclass_accuracy(logits, labels, 
                                         num_classes=self.args.num_labels).detach().cpu().item()
        return acc_score
    
