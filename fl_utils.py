import os
import torch
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
import fl_quant
from peft import LoraConfig, get_peft_model

# PFL_DOCVQA Libraries
from utils import load_config, parse_args, seed_everything, save_yaml
from build_utils import build_model

def add_adapters(model, lora_rank, lora_alpha):
    orig = sum([p.numel() for p in model.parameters() if p.requires_grad])
    if lora_rank > 0:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"], # k,q,v,o
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["visual_embedding.visual_emb_matcher", "spatial_embedding"],
        )
        model = get_peft_model(model, lora_config)
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Training {trainable}/{orig} originally trainable parameters ({100*trainable/orig:.4g}%)")
    return model

from communication.log_communication import log_communication
if __name__ == "__main__":
    seed_everything(0)
    args = parse_args()
    config = load_config(args)
    if args.model_agg:        
        aggregate = None
        ckpt_paths = args.ckpt.split(',')
        print("Averaging checkpoints:", ckpt_paths)
        for ckpt_path in ckpt_paths:
            server_params = torch.load(ckpt_path)
            if aggregate is None:
                aggregate = {n:p.data for n,p in server_params.items()}
            else:
                for n,p in server_params.items():
                    aggregate[n] += fl_quant.quantize_blockwise(p.data) if args.quantize else p.data
        for n,p in aggregate.items():
            aggregate[n] = p / len(ckpt_paths)
            # if args.quantize:
            #     aggregate[n] = fl_quant.quantize_blockwise(aggregate[n])
        torch.save(aggregate, args.name)
    elif args.merge_lora:
        model = build_model(config)
        model.model = add_adapters(model.model, args.lora_rank, args.lora_alpha)
        server_params = torch.load(args.ckpt, map_location='cpu')
        for n,p in model.model.named_parameters():
            if p.requires_grad:
                p.data = server_params[n]
                if args.quantize:
                    p.data = fl_quant.quantize_blockwise(p.data)
        model.model = model.model.merge_and_unload()
        if args.quantize:
            ckpt_name = f"{args.ckpt[:-5]}_quant_merged.ckpt"
        else:
            ckpt_name = f"{args.ckpt[:-5]}_merged.ckpt"
        model.model.save_pretrained(ckpt_name)

        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_name)

        save_yaml(f"{ckpt_name}/experiment_config.yml", config)
    else:
        raise ValueError()
    # elif args.agg_operation == 'agg':
