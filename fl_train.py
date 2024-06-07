import os
import argparse
import torch
import torch.distributed as dist
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
from torch.multiprocessing import Process
from tqdm import tqdm
from torch.utils.data import DataLoader

# PFL_DOCVQA Libraries
from datasets.PFL_DocVQA import collate_fn
from utils import load_config, parse_args, seed_everything
from build_utils import build_dataset, build_model
from metrics import Evaluator
from eval import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model
import copy
import fl_quant
from communication.log_communication import log_communication

DATA = "/projects/smith/kkuo2/pfl_docvqa"
CORES = 24
EVAL_BATCH = 6


# Report standard error using finite population correction + Bessel's correction:
# https://stats.stackexchange.com/questions/546064/why-the-standard-error-of-the-mean-never-gets-to-zero-even-when-the-sample-has-t
def get_stderr(S, n, N):
    stderr = (S.std() / np.sqrt(n)) * np.sqrt((N - n) / N)
    return stderr


def train_model_dataparallel(rank, gpus, model, opt_params, optimizer, epochs, train_data_loader, pbar, pbar_prefix):
    model.model = model.model.cuda(rank)
    step = 0
    # Iterations over the training dataset (for each GPU)
    gpu_batches_per_epoch = len(train_data_loader) // gpus
    for gpu_epoch in range(int(np.ceil(epochs/gpus))):
        for i, batch in enumerate(train_data_loader):
            # Break after len(train_data_loader) batches have been processed globally.
            cpu_epoch = step // gpu_batches_per_epoch
            # if i == 2: # debugging
            if cpu_epoch == epochs:
                break
            step += 1
            # Broadcast and zero gradient
            for n,p in opt_params.items():
                dist.broadcast(p.data, 0)
                p.grad = None
            # Get aggregate gradient
            outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)
            loss = outputs.loss
            loss.backward()
            for n,p in opt_params.items():
                dist.reduce(p.grad, dst=0, op=dist.ReduceOp.SUM)
            # Apply optimizer update
            if rank == 0:
                for n,p in opt_params.items():
                    p.grad /= args.gpus
                optimizer.step()
                # Logging
                global_i = i % gpu_batches_per_epoch
                if i % 100 == 0:
                    desc = f"({pbar_prefix}) Epoch {cpu_epoch}, Batch {global_i}/{gpu_batches_per_epoch}), Loss {loss.item():.2f}"
                    pbar.set_description(desc)
    return opt_params


def eval_model_dataparallel(rank, gpus, model, val_dataset, subsample=1):
    evaluator = Evaluator()
    model.model = model.model.cuda(rank)
    model.model = model.model.eval()
    N_val = len(val_dataset)
    N_val_gpu = int((N_val / gpus)*subsample)
    rand_idx = torch.randperm(N_val) if subsample < 1 else torch.arange(N_val)
    dist.broadcast(rand_idx, 0)
    sub_idx_by_rank = [list(rand_idx[i] for i in range(r*N_val_gpu, min(N_val, (r+1)*N_val_gpu))) for r in range(gpus)]
    val_subset = torch.utils.data.Subset(val_dataset, sub_idx_by_rank[rank])
    val_data_loader = DataLoader(val_subset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn)
    accu = []
    anls = []
    for i, batch in enumerate(val_data_loader):
        with torch.no_grad():
            outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)
            metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))
            accu.extend(metric['accuracy'])
            anls.extend(metric['anls'])
    accu = torch.Tensor(accu)
    anls = torch.Tensor(anls)
    accus = [torch.zeros(len(sub_idx)) for sub_idx in sub_idx_by_rank] if rank == 0 else None
    anlss = [torch.zeros(len(sub_idx)) for sub_idx in sub_idx_by_rank] if rank == 0 else None
    dist.gather(accu, accus, dst=0)
    dist.gather(anls, anlss, dst=0)
    if rank == 0:
        accus = torch.cat(accus)
        anlss = torch.cat(anlss)
    model.model = model.model.train()
    return accus, anlss


def add_adapters(model, lora_rank, lora_alpha):
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
    return model


def run(rank, **kwargs):
    seed_everything(rank)
    args = argparse.Namespace(**kwargs)
    config = load_config(args)
    torch.set_num_threads(CORES // args.gpus)
    print("threads", torch.get_num_threads())
    # SETUP
    val_dataset = build_dataset(config, 'val')
    server_model = build_model(config)
    num_orig = sum([p.numel() for p in server_model.model.parameters() if p.requires_grad])
    server_model.model = add_adapters(server_model.model, args.lora_rank, args.lora_alpha)
    num_trainable = sum([p.numel() for p in server_model.model.parameters() if p.requires_grad])
    server_params = {n:p for n,p in server_model.model.named_parameters() if p.requires_grad}
    if args.ckpt:
        if rank == 0:
            print(f"Loading weights from {args.ckpt}")
        ckpt_params = torch.load(args.ckpt, map_location='cpu')
        for n,p in server_model.model.named_parameters():
            if p.requires_grad:
                p.data = ckpt_params[n]
                if args.quantize:
                    p.data = fl_quant.quantize_blockwise(p.data)
    epochs = config.fl_params.iterations_per_fl_round
    if rank == 0:
        print(f"Training {num_trainable}/{num_orig} originally trainable parameters ({100*num_trainable/num_orig:.4g}%)")
        if args.name:
            writer = tf.summary.create_file_writer(args.name)
        else:
            print("warning: no logging directory specified (args.name). results will not be logged")
    
    if args.eval_ckpt:
        if rank == 0:
            print(f"Running eval on {args.ckpt}")
        accus, anlss = eval_model_dataparallel(rank, args.gpus, server_model, val_dataset)
        if rank == 0:
            print(len(accus), len(val_dataset))
            accu = accus.mean().item()
            accu_std = get_stderr(accus, len(accus), len(val_dataset)).item()
            anls = anlss.mean().item()
            anls_std = get_stderr(anlss, len(anlss), len(val_dataset)).item()
            print(f"Accu (stderr): {accu} ({accu_std}), ANLS (stderr): {anls} ({anls_std})")
        return
    elif args.client_id != -1: # Train args.epochs on a single client
        if rank == 0:
            print(f"Training on client {args.client_id} for {epochs} epochs")
        train_dataset = build_dataset(config, 'train', args.client_id)
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        pbar = tqdm(total=1, disable=rank != 0)
        optimizer = torch.optim.AdamW(list(server_params.values()), lr=float(config.lr)) if rank == 0 else None
        for epoch in range(epochs):
            train_model_dataparallel(
                rank=rank, gpus=args.gpus, 
                model=server_model, opt_params=server_params, optimizer=optimizer, 
                epochs=1, train_data_loader=train_data_loader, 
                pbar=pbar, pbar_prefix=f"Client {args.client_id} Epoch {epoch}")
            if epoch in [0,1,3,7,15]:
                accus, anlss = eval_model_dataparallel(rank, args.gpus, server_model, val_dataset)
                if rank == 0:
                    with writer.as_default():
                        tf.summary.scalar(f"eval/accu", accus.mean().item(), step=epoch)
                        tf.summary.scalar(f"eval/anls", anlss.mean().item(), step=epoch)
                        tf.summary.scalar(f"eval/accu_stderr", get_stderr(accus, len(accus), len(val_dataset)).item(), step=epoch)
                        tf.summary.scalar(f"eval/anls_stderr", get_stderr(anlss, len(anlss), len(val_dataset)).item(), step=epoch)
                    torch.save(server_params, f"{args.name}/client_{args.client_id}-epoch_{epoch}.ckpt")
        return

    # Run FL
    if rank == 0:
        print(f"Running FL for {args.num_rounds} rounds, {args.sample_clients} clients per round, {epochs} local epochs")
    pbar = tqdm(total=args.num_rounds*args.sample_clients*epochs, disable=rank != 0)
    for rnd in range(args.num_rounds):
        # TRAINING
        agg_params = None
        sampled_clients = torch.randperm(10)[:args.sample_clients]
        dist.broadcast(sampled_clients, 0)
        for client_i, client_id in enumerate(sampled_clients):        
            train_dataset = build_dataset(config, 'train', client_id)
            train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
            client_model = copy.deepcopy(server_model)
            client_params = {n:p for n,p in client_model.model.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW(list(client_params.values()), lr=float(config.lr))
            pbar_prefix = f"Round {rnd}, Client {client_id} ({client_i}/{args.sample_clients})"
            train_model_dataparallel(
                rank=rank, gpus=args.gpus, 
                model=client_model, opt_params=client_params, optimizer=optimizer, 
                epochs=epochs, train_data_loader=train_data_loader, 
                pbar=pbar, pbar_prefix=pbar_prefix)
            # AGGREGATION                
            if rank == 0:
                log_communication(federated_round=rnd, sender=-1, receiver=client_id, data=list(server_params.values()), log_location=f"{args.name}/comm_log.csv")
                log_communication(federated_round=rnd, sender=client_id, receiver=-1, data=list(server_params.values()), log_location=f"{args.name}/comm_log.csv")
                if agg_params is None:
                    agg_params = {n:torch.zeros_like(p.data) for n,p in client_params.items()}
                for n,p in client_params.items():
                    if args.quantize:
                        p.data = fl_quant.quantize_blockwise(p.data)
                    agg_params[n] += p.data
        if rank == 0:
            for n,p in agg_params.items():
                agg_params[n] = p / args.sample_clients
                if args.quantize:
                    agg_params[n] = fl_quant.quantize_blockwise(agg_params[n])
                server_params[n].data = agg_params[n]
        for n,p in server_params.items():
            dist.broadcast(p.data, 0)

        # EVALUATION
        if rnd in [0,1,3,7]:
            client_model = copy.deepcopy(server_model)
            accus, anlss = eval_model_dataparallel(rank, args.gpus, client_model, val_dataset)
            if rank == 0:
                with writer.as_default():
                    tf.summary.scalar(f"eval/accu", accus.mean().item(), step=rnd)
                    tf.summary.scalar(f"eval/anls", anlss.mean().item(), step=rnd)
                    tf.summary.scalar(f"eval/accu_stderr", get_stderr(accus, len(accus), len(val_dataset)).item(), step=rnd)
                    tf.summary.scalar(f"eval/anls_stderr", get_stderr(anlss, len(anlss), len(val_dataset)).item(), step=rnd)
                torch.save(server_params, f"{args.name}/round_{rnd}.ckpt")


def init_processes(rank, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=args.gpus)
    fn(rank, **vars(args))


if __name__ == "__main__":
    args = parse_args()
    if args.name:
        os.makedirs(args.name)
    processes = []
    for rank in range(args.gpus):
        p = Process(target=init_processes, args=(rank, run, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()