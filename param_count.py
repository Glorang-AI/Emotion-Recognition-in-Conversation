import argparse
from transformers import Wav2Vec2Config, BertConfig

from models import CASEAttentionModel, CASECompressingModel, MultiModalMixer


if __name__ == "__main__":
    # Define a config dictionary object
    parser = argparse.ArgumentParser()

    # -- Choose Pretrained Model
    parser.add_argument("--lm_path", type=str, default="klue/bert-base", help="You can choose models among (klue-bert series and klue-roberta series) (default: klue/bert-base")
    parser.add_argument("--am_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean")

    # -- Training Argument
    parser.add_argument("--context_max_len", type=int, default=128)
    parser.add_argument("--audio_max_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--pet", type=bool, default=False)

    # -- Model Argument
    parser.add_argument("--opt", type=str, default='mean', help="Can choose operators type between 'mean' and 'sum'")
    parser.add_argument("--mm_type", type=str, default='add', help="concat or add")
    parser.add_argument("--num_labels", type=int, default=7)

    # -- utils
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    wav_config = Wav2Vec2Config.from_pretrained(args.am_path)
    bert_config = BertConfig.from_pretrained(args.lm_path)

    mmm = MultiModalMixer(args, wav_config, bert_config)
    mmm.freeze()

    attention = CASEAttentionModel(args, wav_config, bert_config)
    compressing = CASECompressingModel(args, wav_config, bert_config)

    print("MMM Paramter 수:", sum(p.numel() for p in mmm.parameters() if p.requires_grad))
    print("CASE (compressing) Paramter 수:", sum(p.numel() for p in compressing.parameters() if p.requires_grad))
    print("CASE (attention) 수:", sum(p.numel() for p in attention.parameters() if p.requires_grad))