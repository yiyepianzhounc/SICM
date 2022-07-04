# -*- coding: utf-8 -*-

"""
The training of SICM
"""

import logging
import os
import random
import time

from packaging import version

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.configuration_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from config import get_argparse
from model import SICM, BertFeatureLSTMCRF, BertBiLSTMCRF
from dataset import TaskDataset
from features.lexical.vocab import ItemVocabFile, ItemVocabArray
from utils.evaluate import seq_f1_with_mask
from features.lexical.lexical import build_lexicon_tree_from_vocabs, get_corpus_matched_word_from_lexicon_tree
from utils.utils import build_pretrained_embedding_for_corpus, save_preds_for_seq_labelling

from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logfile = 'data/log/SICM_{}.log'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))

fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(dataset, args, mode='train'):

    print("Dataset length: ", len(dataset))

    if mode == 'train':
        sampler = RandomSampler(dataset)
        batch_size = args.per_gpu_train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.per_gpu_eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=sampler
    )

    return data_loader

def get_optimizer(model, args, num_training_steps):

    no_bigger = ["word_embedding", "attn_w", "word_transform", "word_word_weight",
                 "hidden2tag", "lstm", "crf",
                 "phonetic_transform", "glyph_transform", "phonetic_glyph_transform",
                 "phonetic_transform_512", "glyph_transform_256",
                 "fused_transform", "fused_transform_phonetic", "fused_transform_glyph",
                 "Wg", "Wh", "att_transform", "W", "Wf",
                 "text_linear", "att_linear", "gate_linear", "reserved_linear", "output_linear"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
            "lr": 0.0001
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler


def print_log(logs, epoch, global_step, eval_type, tb_writer, iterator=None):
    if epoch is not None:
        logs['epoch'] = epoch
    if global_step is None:
        global_step = 0
    if eval_type in ["Dev", "Test"]:
        print("#############  %s's result  #############"%(eval_type))
    if tb_writer:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    '"%s" of type %s for key "%s" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.",
                    v,
                    type(v),
                    k,
                )
        tb_writer.flush()

    output = {**logs, **{"step": global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)


def train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer, model_path=None):
    
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    
    if args.max_steps > 0:
        t_total = args.max_steps 
        num_train_epochs = (
                args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1  
        )
    
    else:
        t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)  
        num_train_epochs = args.num_train_epochs
    
    optimizer, scheduler = get_optimizer(model, args, t_total)

    
    if (model_path is not None
        and os.path.isfile(os.path.join(model_path.rstrip("pytorch_model.bin"), "optimizer.pt"))  
        and os.path.isfile(os.path.join(model_path.rstrip("pytorch_model.bin"), "scheduler.pt"))  
    ):
        optimizer.load_state_dict(
            torch.load(os.path.join(model_path.rstrip("pytorch_model.bin"), "optimizer.pt"), map_location=args.device)  
        )
        scheduler.load_state_dict(torch.load(os.path.join(model_path.rstrip("pytorch_model.bin"), "scheduler.pt")))

    model = model.cuda()  
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )
    
    total_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps  

    logger.info("***** Start training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))  
    logger.info("  Num Epochs = %d", num_train_epochs)                 
    logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)  
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)  
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  
    logger.info("  Total optimization steps = %d", t_total)  

    global_step = 0
    epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    
    if model_path is not None: 
        try:
            global_step = int(model_path.split("-")[-1].split("/")[0])  
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
            )  
            model.load_state_dict(torch.load(model_path))  
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)  
        except ValueError:
            global_step = 0
            logger.info("  Starting fine-tuning.")  

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()  
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    
    for epoch in train_iterator:
        
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler):
            train_dataloader.sampler.set_epoch(epoch)  
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")  

        
        
        for step, batch in enumerate(epoch_iterator):
            
            if steps_trained_in_current_epoch > 0:
                
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8])
            new_batch = batch_data
            
            batch = tuple(t.to(args.device) for t in new_batch)
            
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "matched_word_ids": batch[3], "matched_word_mask": batch[4],
                      "boundary_ids": batch[5], "labels": batch[6],
                      "phonetic_features": batch[7], "glyph_features": batch[8],"flag": "Train"}
            batch_data = None
            new_batch = None

            outputs = model(**inputs)  
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps  

            else:
                loss.backward()  

            tr_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    ((step + 1) == len(epoch_iterator)):

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()  

                scheduler.step()  
                model.zero_grad()  
                global_step += 1
                
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    logs["loss"] = (tr_loss - logging_loss) / args.logging_steps  
                    
                    logs["learning_rate"] = (
                        scheduler.get_last_lr()[0]
                        if version.parse(torch.__version__) >= version.parse("1.4")
                        else scheduler.get_lr()[0]
                    )
                    logging_loss = tr_loss  
                    print_log(logs, epoch, global_step, "", tb_writer)
                
                if False and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")  
                    os.makedirs(output_dir, exist_ok=True)
                    
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    if False and args.evaluate_during_training:
                        metrics, _ = evaluate(
                            model, args, test_dataset, label_vocab, global_step, description="Test")
                        print_log(metrics, epoch, global_step, "Test", tb_writer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if args.evaluate_during_training:  

            metrics, _ = evaluate(model, args, test_dataset, label_vocab, global_step, description="Test", write_file=True)
            print_log(metrics, epoch, global_step, "Test", tb_writer)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
    os.makedirs(output_dir, exist_ok=True)  

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    print("global_step: ", global_step)  
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    
    return global_step, tr_loss / global_step

def evaluate(model, args, dataset, label_vocab, global_step, description="dev", write_file=False):

    dataloader = get_dataloader(dataset, args, mode='dev')

    if (not args.do_train) and (not args.no_cuda):
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", batch_size)
    eval_losses = []

    model.eval()

    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_attention_mask = None

    for batch in tqdm(dataloader, desc=description):

        batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8])
        new_batch = batch_data
        batch = tuple(t.to(args.device) for t in new_batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                  "matched_word_ids": batch[3], "matched_word_mask": batch[4],
                  "boundary_ids": batch[5], "labels": batch[6],
                  "phonetic_features": batch[7], "glyph_features": batch[8], "flag": "Predict"}
        batch_data = None
        new_batch = None

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[6].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()

        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_attention_mask = attention_mask
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)  
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)  
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)  
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
    
    acc, p, r, f1, all_true_labels, all_pred_labels = seq_f1_with_mask(
        all_label_ids, all_predict_ids, all_attention_mask, label_vocab)
    metrics = {}
    metrics['acc'] = acc
    metrics['p'] = p
    metrics['r'] = r
    metrics['f1'] = f1
    
    if write_file:
        file_path = os.path.join(args.output_dir, "{}-{}-{}.txt".format(args.model_type, description, str(global_step)))  
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
        
        save_preds_for_seq_labelling(all_input_ids, tokenizer, all_true_labels, all_pred_labels, file_path)

    return metrics, (all_true_labels, all_pred_labels)


def main():
    args = get_argparse().parse_args()

    device = torch.device("cuda:0" if not args.no_cuda else "cpu")  
    print(device)
    torch.cuda.set_device(device)
    args.device = device  

    logger.info("Training/evaluation parameters %s", args)
    
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
    
    set_seed(args.seed)
    
    print("Build lexicon tree...")
    lexicon_tree = build_lexicon_tree_from_vocabs([args.word_vocab_file], scan_nums=[args.max_scan_num])  
    embed_lexicon_tree = lexicon_tree
    
    train_data_file = os.path.join(args.data_dir, "train.json")
    print(train_data_file)
    
    if "msra" in args.data_dir:
        dev_data_file = os.path.join(args.data_dir, "test.json")
    else:
        dev_data_file = os.path.join(args.data_dir, "dev.json")

    test_data_file = os.path.join(args.data_dir, "test.json")
    data_files = [train_data_file, dev_data_file, test_data_file]  

    print("Get matched words...\n")
    matched_words = get_corpus_matched_word_from_lexicon_tree(data_files, embed_lexicon_tree)  
    word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)  
    label_vocab = ItemVocabFile(files=[args.label_file], is_word=False)  

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    with open("word_vocab.txt", "w", encoding="utf-8") as f:
        for idx, word in enumerate(word_vocab.idx2item):
            f.write("%d\t%s\n"%(idx, word))
    
    print("Build pretrained embedding for corpus...")
    pretrained_word_embedding, embed_dim = build_pretrained_embedding_for_corpus(
        embedding_path=args.word_embedding,
        word_vocab=word_vocab,
        embed_dim=args.word_embed_dim,
        max_scan_num=args.max_scan_num,
        saved_corpus_embedding_dir=args.saved_embedding_dir,
    )
    
    config = BertConfig.from_pretrained(args.config_name)
    config.add_phonetic = args.add_phonetic
    config.add_glyph = args.add_glyph
    config.add_feature_gate = args.add_feature_gate
    config.add_final_attention = args.add_final_attention
    config.add_final_gate = args.add_final_gate
    config.add_middle_attention = args.add_middle_attention
    print(config.add_phonetic, config.add_glyph, config.add_final_attention,
          config.add_feature_gate,  config.add_final_gate, config.add_middle_attention)
    
    if args.model_type == "SICM":                  
        model = SICM.from_pretrained(
            args.model_name_or_path,                          
            config=config,                                    
            pretrained_embeddings=pretrained_word_embedding,  
            num_labels=label_vocab.get_item_size())           
    elif args.model_type == "BertFeatureLSTMCRF":
        model = BertFeatureLSTMCRF.from_pretrained(
            args.model_name_or_path,
            config=config,
            pretrained_embeddings=pretrained_word_embedding,
            num_labels=label_vocab.get_item_size())
    elif args.model_type == "BertBiLSTMCRF":
        model = BertBiLSTMCRF.from_pretrained(
            args.model_name_or_path,
            config=config,
            num_labels=label_vocab.get_item_size()
        )

    if not args.no_cuda:
        model = model.cuda()

    args.label_size = label_vocab.get_item_size()
    dataset_params = {
        'tokenizer': tokenizer,                  
        'word_vocab': word_vocab,                
        'label_vocab': label_vocab,              
        'lexicon_tree': lexicon_tree,            
        'max_seq_length': args.max_seq_length,
        'max_scan_num': args.max_scan_num,
        'max_word_num': args.max_word_num,       
        'default_label': args.default_label,     
    }

    if args.do_train:
        
        train_dataset = TaskDataset(train_data_file, params=dataset_params, do_shuffle=args.do_shuffle)  
        dev_dataset = TaskDataset(dev_data_file, params=dataset_params, do_shuffle=False)
        test_dataset = TaskDataset(test_data_file, params=dataset_params, do_shuffle=False)

        train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer, model_path=None)  

    if args.do_eval:
        logger.info("*** Dev Evaluate ***")
        dev_dataset = TaskDataset(dev_data_file, params=dataset_params, do_shuffle=False)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]  
        
        eval_output, _ = evaluate(model, args, dev_dataset, label_vocab, global_steps, "dev", write_file=True)
        eval_output["global_steps"] = global_steps
        print("Dev Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n"%
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))

    if args.do_predict:
        logger.info("*** Test Evaluate ***")
        test_dataset = TaskDataset(test_data_file, params=dataset_params, do_shuffle=False)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]
        
        eval_output, _ = evaluate(model, args, test_dataset, label_vocab, global_steps, "test", write_file=True)
        eval_output["global_steps"] = global_steps
        print("Test Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))

if __name__ == "__main__":
    main()
