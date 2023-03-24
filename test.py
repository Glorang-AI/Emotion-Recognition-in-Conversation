import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Wav2Vec2Config, 
    BertConfig,
    AutoTokenizer
)

from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from tqdm import tqdm

from dataset import ETRIDataset
from models import CASEmodel, RoCASEmodel, CompressedCCEModel, ConcatModel, MultiModalMixer
from utils import audio_embedding, seed

def test(args):
    seed.seed_setting(args.seed)

    wav_config = Wav2Vec2Config.from_pretrained(args.am_path)
    bert_config = BertConfig.from_pretrained(args.lm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_path)

    def text_audio_collator(batch):
       
        return {'audio_emb' : pad_sequence([item['audio_emb'] for item in batch], batch_first=True),
                'label' : torch.stack([item['label'] for item in batch]).squeeze(),
                'input_ids' :  torch.stack([item['input_ids'] for item in batch]).squeeze(),
                'attention_mask' :  torch.stack([item['attention_mask'] for item in batch]).squeeze(),
                'token_type_ids' :  torch.stack([item['token_type_ids'] for item in batch]).squeeze()}

    # args.data_path
    # args.embedding_path
    test_data = pd.read_csv(args.data_path)
    test_data.reset_index(inplace=True)

    audio_emb = audio_embedding.save_and_load(args.am_path, test_data['audio'].tolist(), args.device, args.embedding_path)

    label_dict = {'angry':0, 'neutral':1, 'sad':2, 'happy':3, 'disqust':4, 'surprise':5, 'fear':6}
    test_dataset = ETRIDataset(
        audio_embedding = audio_emb, 
        dataset=test_data, 
        label_dict = label_dict,
        tokenizer = tokenizer,
        audio_emb_type = args.audio_emb_type,
        max_len = args.context_max_len, 
        )

    # Create a DataLoader that batches audio sequences and pads them to a fixed length
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.test_bsz,
        shuffle=False, 
        collate_fn=text_audio_collator, 
        num_workers=args.num_workers,
        )
    
    if args.model == "CASE":
        model = CASEmodel(args.lm_path, wav_config, bert_config, args.num_labels)
    elif args.model == "CCE":
        model = CompressedCCEModel(args, wav_config, bert_config)
    elif args.model == "Concat":
        model = ConcatModel(args, wav_config, bert_config)
    elif args.model == "MMM":
        model = MultiModalMixer(args, wav_config, bert_config)
        model.freeze()

    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    model.eval()

    test_output = []
    test_label = []
    with torch.no_grad():

        pbar = tqdm(test_dataloader)
        for _, batch in enumerate(pbar):
            label = batch['label'].to(args.device)
            audio_tensor = batch['audio_emb'].to(args.device)

            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            token_type_ids = batch["token_type_ids"].to(args.device)

            output = model(
                input_ids, 
                attention_mask,
                token_type_ids,
                audio_tensor 
                )['class_logit']
            
            test_output.append(output.detach().cpu())
            test_label.append(label.detach().cpu())

        logits = torch.cat(test_output)
        labels = torch.cat(test_label)

        test_m_f1 = multiclass_f1_score(logits, labels, 
                                         num_classes=args.num_labels, 
                                         average="macro").detach().cpu().item()
        test_w_f1 = multiclass_f1_score(logits, labels, 
                                         num_classes=args.num_labels, 
                                         average="weighted").detach().cpu().item()
        test_acc = multiclass_accuracy(logits, labels, 
                                         num_classes=args.num_labels).detach().cpu().item()
    
        print(f'Macro F1 Score: {test_m_f1}, Weighted F1 Score: {test_w_f1}, Accuracy: {test_acc}')

if __name__ == "__main__":
        # Define a config dictionary object
    parser = argparse.ArgumentParser()

    # -- Choose Pretrained Model
    parser.add_argument("--lm_path", type=str, default="klue/bert-base", help="You can choose models among (klue-bert series and klue-roberta series) (default: klue/bert-base")
    parser.add_argument("--am_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean")

    # -- Training Argument
    parser.add_argument("--test_bsz", type=int, default=256)
    parser.add_argument("--context_max_len", type=int, default=128)
    parser.add_argument("--audio_max_len", type=int, default=1024)
    parser.add_argument("--num_labels", type=int, default=7)
    parser.add_argument("--audio_emb_type", type=str, default="last_hidden_state", help="Can chosse audio embedding type between 'last_hidden_state' and 'extract_features' (default: last_hidden_state)")
    parser.add_argument("--model", type=str, default="CASE")

    ## -- directory
    parser.add_argument("--data_path", type=str, default="data/test.csv")
    parser.add_argument("--model_path", type=str, default="save/epoch:1_CASEmodel.pt")
    ###### emb_train에 대한 설명 부과하기
    parser.add_argument("--embedding_path", type=str, default="data/emb_test.pt")

    # -- utils
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    test(args)
