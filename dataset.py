# -*- coding: utf-8 -*-

import os
import json
import random
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from features.lexical.lexical import sent_to_matched_words_boundaries
from features.phonetic import build_pinyin_map, txt2pinyin
from features.glyph import build_glyph_dict, txt2stroke

random.seed(106524)

class TaskDataset(Dataset):
    def __init__(self, file, params, do_shuffle=False):

        self.max_word_num = params['max_word_num']
        self.tokenizer = params['tokenizer']
        self.label_vocab = params['label_vocab']
        self.word_vocab = params['word_vocab']
        self.lexicon_tree = params['lexicon_tree']
        self.max_scan_num = params['max_scan_num']
        self.max_seq_length = params['max_seq_length']
        self.default_label = params['default_label']
        self.do_shuffle = do_shuffle

        self.phonetic_dim = 25
        self.glyph_dim = 25

        self.file = file  
        file_items = file.split("/")  
        data_dir = "/".join(file_items[:-1])  

        file_name = "saved_maxword_{}_maxseq_{}_".format(self.max_word_num, self.max_seq_length) + \
                    file_items[-1].split('.')[0] + "_features" + "_{}.npz".format(self.max_scan_num)  
        saved_np_file = os.path.join(data_dir, file_name)  
        self.np_file = saved_np_file  

        self.init_np_dataset()

    def init_np_dataset(self):

        print_flag = True
        
        if os.path.exists(self.np_file):
            with np.load(self.np_file) as dataset:  
                
                self.input_ids = dataset["input_ids"]
                self.segment_ids = dataset["segment_ids"]
                self.attention_mask = dataset["attention_mask"]
                self.input_matched_word_ids = dataset["input_matched_word_ids"]
                self.input_matched_word_mask = dataset["input_matched_word_mask"]
                self.input_boundary_ids = dataset["input_boundary_ids"]
                self.labels = dataset["labels"]

                self.phonetic_features = dataset["phonetic_features"]
                self.glyph_features = dataset["glyph_features"]

            print("核对%s中id和词是否匹配: "%(self.file))
            print(self.input_ids[0][:10])  
            print(self.tokenizer.convert_ids_to_tokens(self.input_ids[0][:10]))  
            for idx in range(10):
                print(self.input_matched_word_ids[0][idx])  
                print(self.word_vocab.convert_ids_to_items(self.input_matched_word_ids[0][idx]))  
            for idx in range(10):
                print(self.phonetic_features[0][idx])
                print(self.glyph_features[0][idx])
        
        else:
            all_input_ids = []
            all_segment_ids = []
            all_attention_mask = []
            all_input_matched_word_ids = []
            all_input_matched_word_mask = []
            all_input_boundary_ids = []
            all_labels = []

            all_phonetic_features = []
            all_glyph_features = []
            
            with open(self.file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)  
                        text = sample['text']  
                        label = sample['label']  
                        if len(text) > self.max_seq_length - 2:
                            text = text[:self.max_seq_length-2]  
                            label = label[:self.max_seq_length-2]
                        text.insert(0, '[CLS]')  
                        label.insert(0, self.default_label)  
                        text.append('[SEP]')  
                        label.append(self.default_label)  

                        token_ids = self.tokenizer.convert_tokens_to_ids(text)  
                        label_ids = self.label_vocab.convert_items_to_ids(label)  

                        input_ids = np.zeros(self.max_seq_length, dtype=np.int_)
                        segment_ids = np.ones(self.max_seq_length, dtype=np.int_)  
                        attention_mask = np.zeros(self.max_seq_length, dtype=np.int_)
                        matched_word_ids = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int_)  
                        matched_word_mask = np.zeros((self.max_seq_length, self.max_word_num), dtype=np.int_)
                        boundary_ids = np.zeros(self.max_seq_length, dtype=np.int_)

                        
                        np_label = np.zeros(self.max_seq_length, dtype=np.int_)
                        np_label[:len(label_ids)] = label_ids

                        phonetic_features = np.zeros((self.max_seq_length, self.phonetic_dim), dtype=np.int_)
                        glyph_features = np.zeros((self.max_seq_length, self.phonetic_dim), dtype=np.int_)
                        
                        str_text = "".join(text)
                        aux_map, vow_map = build_pinyin_map()
                        glyph_dict = build_glyph_dict()
                        try:
                            phonetic_features[:len(token_ids)] = txt2pinyin(str_text, aux_map, vow_map)
                        except ValueError:
                            print(str_text, text)
                        glyph_features[:len(token_ids)] = txt2stroke(str_text, glyph_dict)
                        
                        input_ids[:len(token_ids)] = token_ids  
                        segment_ids[:len(token_ids)] = 0  
                        attention_mask[:len(token_ids)] = 1
                        
                        matched_words, sent_boundaries = \
                            sent_to_matched_words_boundaries(text, self.lexicon_tree, self.max_word_num)  
                        sent_length = len(text)  
                        boundary_ids[:len(sent_boundaries)] = sent_boundaries  
                        for idy in range(sent_length):
                            now_words = matched_words[idy]  
                            now_word_ids = self.word_vocab.convert_items_to_ids(now_words)  
                            matched_word_ids[idy][:len(now_word_ids)] = now_word_ids  
                            matched_word_mask[idy][:len(now_word_ids)] = 1  

                        if print_flag:
                            print("核对%s中id和词是否匹配: "%(self.file))
                            print(input_ids[:10])
                            print(self.tokenizer.convert_ids_to_tokens(input_ids[:10]))
                            for idx in range(10):
                                print(matched_word_ids[idx])
                                print(self.word_vocab.convert_ids_to_items(matched_word_ids[idx]))
                            for idx in range(10):
                                print(phonetic_features[idx])
                                print(glyph_features[idx])

                            print(matched_words)
                            print(matched_words[:10])
                            print(matched_word_ids[:10])  
                            print_flag = False  

                        all_input_ids.append(input_ids)
                        all_segment_ids.append(segment_ids)
                        all_attention_mask.append(attention_mask)
                        all_input_matched_word_ids.append(matched_word_ids)  
                        all_input_matched_word_mask.append(matched_word_mask)  
                        all_input_boundary_ids.append(boundary_ids)  
                        all_labels.append(np_label)  

                        all_phonetic_features.append(phonetic_features)  
                        all_glyph_features.append(glyph_features)  

            assert len(all_input_ids) == len(all_segment_ids), (len(all_input_ids), len(all_segment_ids))  
            assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
            assert len(all_input_ids) == len(all_input_matched_word_ids), (len(all_input_ids), len(all_input_matched_word_ids))
            assert len(all_input_ids) == len(all_input_matched_word_mask), (len(all_input_ids), len(all_input_matched_word_mask))
            assert len(all_input_ids) == len(all_input_boundary_ids), (len(all_input_ids), len(all_input_boundary_ids))
            assert len(all_input_ids) == len(all_labels), (len(all_input_ids), len(all_labels))

            assert len(all_input_ids) == len(all_phonetic_features), (len(all_input_ids), len(all_phonetic_features))
            assert len(all_input_ids) == len(all_glyph_features), (len(all_input_ids), len(all_glyph_features))


            all_input_ids = np.array(all_input_ids)
            all_segment_ids = np.array(all_segment_ids)
            all_attention_mask = np.array(all_attention_mask)
            all_input_matched_word_ids = np.array(all_input_matched_word_ids)
            all_input_matched_word_mask = np.array(all_input_matched_word_mask)
            all_input_boundary_ids = np.array(all_input_boundary_ids)
            all_labels = np.array(all_labels)

            all_phonetic_features = np.array(all_phonetic_features)
            all_glyph_features = np.array(all_glyph_features)

            np.savez(
                self.np_file, input_ids=all_input_ids, segment_ids=all_segment_ids, attention_mask=all_attention_mask,
                input_matched_word_ids=all_input_matched_word_ids, input_matched_word_mask=all_input_matched_word_mask,
                input_boundary_ids=all_input_boundary_ids, labels=all_labels, phonetic_features=all_phonetic_features,
                glyph_features = all_glyph_features
            )

            self.input_ids = all_input_ids
            self.segment_ids = all_segment_ids
            self.attention_mask = all_attention_mask
            self.input_matched_word_ids = all_input_matched_word_ids
            self.input_matched_word_mask = all_input_matched_word_mask
            self.input_boundary_ids = all_input_boundary_ids
            self.labels = all_labels

            self.phonetic_features = all_phonetic_features
            self.glyph_features = all_glyph_features

        self.total_size = self.input_ids.shape[0]  
        self.indexes = list(range(self.total_size))  
        if self.do_shuffle:  
            random.shuffle(self.indexes)  

    
    def __len__(self):
        return self.total_size

    
    def __getitem__(self, index):
        index = self.indexes[index]
        return (
            torch.tensor(self.input_ids[index]),                
            torch.tensor(self.segment_ids[index]),              
            torch.tensor(self.attention_mask[index]),           
            torch.tensor(self.input_matched_word_ids[index]),   
            torch.tensor(self.input_matched_word_mask[index]),  
            torch.tensor(self.input_boundary_ids[index]),       
            torch.tensor(self.labels[index]),                   

            torch.tensor(self.phonetic_features[index]),
            torch.tensor(self.glyph_features[index])
        )





