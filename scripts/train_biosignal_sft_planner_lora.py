#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open() as handle:
        for line in handle:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts=[]
        for msg in messages:
            parts.append(f"<{msg.get('role','user')}>\n{msg.get('content','')}\n")
        return ''.join(parts)


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng=random.Random(seed)
    rows=list(rows); rng.shuffle(rows)
    n_val=max(1, int(round(len(rows)*val_fraction))) if len(rows)>1 else 0
    return rows[n_val:], rows[:n_val]


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.items=[]
        for row in rows:
            text=format_messages(tokenizer, row['messages'])
            enc=tokenizer(text, truncation=True, max_length=max_length, padding=False)
            self.items.append({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']})
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def collate(batch, tokenizer):
    max_len=max(len(x['input_ids']) for x in batch)
    pad_id=tokenizer.pad_token_id
    input_ids=[]; masks=[]; labels=[]
    for item in batch:
        ids=item['input_ids']; mask=item['attention_mask']
        pad=max_len-len(ids)
        padded=ids+[pad_id]*pad
        padded_mask=mask+[0]*pad
        lab=padded.copy()
        lab=[tok if m else -100 for tok,m in zip(lab,padded_mask)]
        input_ids.append(padded); masks.append(padded_mask); labels.append(lab)
    return {'input_ids': torch.tensor(input_ids,dtype=torch.long), 'attention_mask': torch.tensor(masks,dtype=torch.long), 'labels': torch.tensor(labels,dtype=torch.long)}


def evaluate_loss(model, loader, device):
    model.eval(); total=0.0; n=0
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(**batch)
            bs=batch['input_ids'].shape[0]
            total += float(out.loss.item())*bs; n += bs
    model.train()
    return total/max(1,n)


def main() -> None:
    ap=argparse.ArgumentParser(description='Train a LoRA SFT planner for BioSignalAgent tool-use JSON planning.')
    ap.add_argument('--train-jsonl', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_v1.jsonl')
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--out-dir', default='/data1/jiahui/biosignal-agent/outputs/biosignal_sft_planner_lora_qwen25_05b')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup-ratio', type=float, default=0.05)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--val-fraction', type=float, default=0.12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--load-4bit', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows=read_jsonl(args.train_jsonl)
    train_rows,val_rows=split_rows(rows,args.val_fraction,args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    quant=None
    if args.load_4bit:
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model_kwargs={'trust_remote_code': True, 'quantization_config': quant, 'torch_dtype': torch.bfloat16 if device.type=='cuda' else torch.float32}
    if args.load_4bit and device.type=='cuda':
        model_kwargs['device_map']='auto'
    model=AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_4bit:
        model=prepare_model_for_kbit_training(model)
    elif device.type=='cuda':
        model.to(device)
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=target_modules)
    model=get_peft_model(model,lora)
    if device.type!='cuda':
        model.to(device)
    if hasattr(model, 'config'):
        model.config.use_cache = False
    model.print_trainable_parameters()
    train_ds=ChatDataset(train_rows,tokenizer,args.max_length)
    val_ds=ChatDataset(val_rows,tokenizer,args.max_length)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=lambda b: collate(b,tokenizer))
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=lambda b: collate(b,tokenizer))
    steps_per_epoch=math.ceil(len(train_loader)/max(1,args.grad_accum))
    total_steps=max(1,steps_per_epoch*args.epochs)
    warmup_steps=int(total_steps*args.warmup_ratio)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sched=get_linear_schedule_with_warmup(opt,warmup_steps,total_steps)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    best_val=None; global_step=0; history=[]
    model.train(); opt.zero_grad(set_to_none=True)
    for epoch in range(1,args.epochs+1):
        total=0.0; count=0
        for i,batch in enumerate(train_loader,1):
            batch={k:v.to(model.device if hasattr(model,'device') else device) for k,v in batch.items()}
            out=model(**batch)
            loss=out.loss/args.grad_accum
            loss.backward()
            total += float(out.loss.item())*batch['input_ids'].shape[0]; count += batch['input_ids'].shape[0]
            if i % args.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
        val_loss=evaluate_loss(model,val_loader,model.device if hasattr(model,'device') else device) if len(val_ds) else None
        rec={'epoch':epoch,'train_loss':total/max(1,count),'val_loss':val_loss,'global_step':global_step}
        history.append(rec); print(json.dumps(rec),flush=True)
        if val_loss is None or best_val is None or val_loss < best_val:
            best_val=val_loss
            model.save_pretrained(out_dir/'best_adapter')
            tokenizer.save_pretrained(out_dir/'best_adapter')
    model.save_pretrained(out_dir/'final_adapter')
    tokenizer.save_pretrained(out_dir/'final_adapter')
    summary={'artifact':'BioSignalSFTPlannerLoRA','base_model':args.base_model,'train_jsonl':args.train_jsonl,'num_examples':len(rows),'num_train':len(train_rows),'num_val':len(val_rows),'epochs':args.epochs,'batch_size':args.batch_size,'grad_accum':args.grad_accum,'lr':args.lr,'max_length':args.max_length,'best_val_loss':best_val,'history':history,'best_adapter':str(out_dir/'best_adapter'),'final_adapter':str(out_dir/'final_adapter')}
    (out_dir/'train_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
