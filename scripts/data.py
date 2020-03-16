import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, split, tokenizer, bwd=False, prefix=None):
        assert split in ('train', 'dev', 'test')
        self.split = split
        self.question_list = os.listdir('data/%s/question' % split)
        self.tokenizer = tokenizer
        self.bwd = bwd
        if prefix:
             self.question_list = [q for q in self.question_list if q.startswith(prefix)]
    
    def __len__(self):
        return len(self.question_list)
        
    def __getitem__(self, i):
        question_id = self.question_list[i]
        with open('data/%s/passage/%s' % (self.split, '|'.join(question_id.split('|')[:2]))) as f:
            passage = f.read().split(' ')
        with open('data/%s/passage_no_unk/%s' % (self.split, '|'.join(question_id.split('|')[:2]))) as f:
            passage_no_unk = f.read().split(' ')

        with open('data/%s/question/%s' % (self.split, question_id)) as f:
            question = f.read().split(' ')
            question.insert(0, self.tokenizer.cls_token)
            question.append(self.tokenizer.sep_token)
        with open('data/%s/question_no_unk/%s' % (self.split, question_id)) as f:
            question_no_unk = f.read().split(' ')
            question_no_unk.insert(0, self.tokenizer.cls_token)
            question_no_unk.append(self.tokenizer.sep_token)

        with open('data/%s/answer/%s' % (self.split, question_id)) as f:
            answer = [line.strip() for line in f]
        with open('data/%s/span/%s' % (self.split, question_id)) as f:
            span = f.read().split(' ')
            answer_start = int(span[0]) + len(question)
            answer_end = int(span[1]) + len(question)
        
        # Truncate length to 512
        diff = len(question) + len(passage) - 511
        if diff > 0:
            if self.split == 'train':
                if answer_end > 510:
                    passage = passage[diff:]
                    passage_no_unk = passage_no_unk[diff:]
                    answer_start -= diff
                    answer_end -= diff
                else:
                    passage = passage[:-diff]
                    passage_no_unk = passage_no_unk[:-diff]
            else:
                if diff > 0:
                    if self.bwd:
                        passage = passage[diff:]
                        passage_no_unk = passage_no_unk[diff:]
                    else:
                        passage = passage[:-diff]
                        passage_no_unk = passage_no_unk[:-diff]

        passage.append(self.tokenizer.sep_token)
        passage_no_unk.append(self.tokenizer.sep_token)
        input_tokens = question + passage
        input_tokens_no_unk = question_no_unk + passage_no_unk
        
        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        attention_mask = torch.FloatTensor([1 for _ in input_tokens])
        token_type_ids = torch.LongTensor([0 for _ in question] + [1 for _ in passage])
        if self.split == 'train':
            start_positions = torch.LongTensor([answer_start]).squeeze(0)
            end_positions = torch.LongTensor([answer_end]).squeeze(0)
            return input_ids, attention_mask, token_type_ids, start_positions, end_positions
        else:
            margin_mask = torch.FloatTensor([*(-1e10 for _ in question), *(0. for _ in passage[:-1]), -1e-10])
            return input_ids, attention_mask, token_type_ids, margin_mask, input_tokens_no_unk, answer
        
class TestBertDataset(Dataset):
    def __init__(self, split, tokenizer, bwd=False, prefix=None):
        assert split in ('dev', 'test')
        self.split = split
        self.question_list = os.listdir('data/%s/question' % split)
        self.tokenizer = tokenizer
        self.bwd = bwd
        if prefix:
             self.question_list = [q for q in self.question_list if q.startswith(prefix)]
    
    def __len__(self):
        return len(self.question_list)
        
    def __getitem__(self, i):
        question_id = self.question_list[i]
        with open('data/%s/passage/%s' % (self.split, '|'.join(question_id.split('|')[:2]))) as f:
            passage = f.read()
        with open('data/%s/question/%s' % (self.split, question_id)) as f:
            question = f.read()
        with open('data/%s/answer/%s' % (self.split, question_id)) as f:
            answer = [line.strip() for line in f]
        return passage, question, answer

class XLNetDataset(Dataset):
    def __init__(self, split, tokenizer, bwd=False, prefix=None):
        assert split in ('train', 'dev', 'test')
        self.split = split
        self.question_list = os.listdir('data/%s/question' % split)
        self.tokenizer = tokenizer
        self.bwd = bwd
        if prefix:
             self.question_list = [q for q in self.question_list if q.startswith(prefix)]

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, i):
        question_id = self.question_list[i]
        with open('data/%s/passage/%s' % (self.split, '|'.join(question_id.split('|')[:2]))) as f:
            passage = f.read().split(' ')
        with open('data/%s/passage_no_unk/%s' % (self.split, '|'.join(question_id.split('|')[:2]))) as f:
            passage_no_unk = f.read().split(' ')

        with open('data/%s/question/%s' % (self.split, question_id)) as f:
            question = f.read().split(' ')
            question.append(self.tokenizer.sep_token)
            question.append(self.tokenizer.cls_token)
        with open('data/%s/question_no_unk/%s' % (self.split, question_id)) as f:
            question_no_unk = f.read().split(' ')
            question_no_unk.append(self.tokenizer.sep_token)
            question_no_unk.append(self.tokenizer.cls_token)

        with open('data/%s/answer/%s' % (self.split, question_id)) as f:
            answer = [line.strip() for line in f]
        with open('data/%s/span/%s' % (self.split, question_id)) as f:
            span = f.read().split(' ')
            answer_start = int(span[0])
            answer_end = int(span[1])
            
        # Truncate length to 512
        diff = len(question) + len(passage) - 511
        if diff > 0:
            if self.split == 'train':
                if answer_end > 510:
                    passage = passage[diff:]
                    passage_no_unk = passage_no_unk[diff:]
                    answer_start -= diff
                    answer_end -= diff
                else:
                    passage = passage[:-diff]
                    passage_no_unk = passage_no_unk[:-diff]
            else:
                if diff > 0:
                    if self.bwd:
                        passage = passage[diff:]
                        passage_no_unk = passage_no_unk[diff:]
                    else:
                        passage = passage[:-diff]
                        passage_no_unk = passage_no_unk[:-diff]

        passage.append(self.tokenizer.sep_token)
        passage_no_unk.append(self.tokenizer.sep_token)
        input_tokens = passage + question
        input_tokens_no_unk = passage_no_unk + question_no_unk

        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        attention_mask = torch.FloatTensor([1 for _ in input_tokens])
        token_type_ids = torch.LongTensor([0 for _ in passage] + [1 for _ in question])
        if self.split == 'train':
            start_positions = torch.LongTensor([answer_start]).squeeze(0)
            end_positions = torch.LongTensor([answer_end]).squeeze(0)
            return input_ids, attention_mask, token_type_ids, start_positions, end_positions
        else:
            return input_ids, attention_mask, token_type_ids, input_tokens_no_unk, answer


    
def get_dataloader(model_type, split, tokenizer, bwd=False, batch_size=1, num_workers=0, prefix=None):
    
    def train_collate_fn(batch):
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
        start_positions = torch.stack(start_positions)
        end_positions = torch.stack(end_positions)
        return input_ids, attention_mask, token_type_ids, start_positions, end_positions
    
    def test_collate_fn(batch):
        return batch
    
    assert model_type in ('bert', 'xlnet')
    shuffle = split == 'train'
    collate_fn = train_collate_fn if split == 'train' else test_collate_fn
    if model_type == 'bert':
        if split == 'train':
            dataset = BertDataset(split, tokenizer, bwd, prefix)
        else:
            dataset = TestBertDataset(split, tokenizer, bwd, prefix)
    elif model_type == 'xlnet':
        dataset = XLNetDataset(split, tokenizer, bwd, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader
