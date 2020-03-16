import sys
import numpy as np
from os.path import join
from copy import deepcopy

import torch
from torch.nn.functional import softmax
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForQuestionAnswering

from utils import AdamW
from data import get_dataloader
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

from datedur import DateDurationQA

np.random.seed(42)
torch.manual_seed(42)

norm_tokenizer = BertTokenizer.from_pretrained('/home/M10815022/Models/bert-wwm-ext')


def validate_dataset(model, split, tokenizer, device, topk=1, prefix=None):
    assert split in ('dev', 'test')
    dataloader = get_dataloader('bert', split, tokenizer, bwd=False, \
                        batch_size=1, num_workers=16, prefix=prefix)
    em, f1, count = 0, 0, 0
    
    ddqa = DateDurationQA(tokenizer, model, device)
    for batch in dataloader:
        passage, question, answer = batch[0]
        preds, _ = ddqa.direct_predict(passage, question, topk=topk)

        count += 1
        if len(preds) > 0:
            norm_preds_tokens = [norm_tokenizer.basic_tokenizer.tokenize(pred) for pred in preds]
            norm_preds = [norm_tokenizer.convert_tokens_to_string(norm_pred_tokens) for norm_pred_tokens in norm_preds_tokens]
            norm_answer_tokens = [norm_tokenizer.basic_tokenizer.tokenize(ans) for ans in answer]
            norm_answer = [norm_tokenizer.convert_tokens_to_string(ans_tokens) for ans_tokens in norm_answer_tokens]

            em += max(metric_max_over_ground_truths(exact_match_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            f1 += max(metric_max_over_ground_truths(f1_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            
    ddqa.tokenizer = None
    ddqa.model = None
    ddqa.device = None
    del ddqa, dataloader
    return em, f1, count

def validate(model, tokenizer, device, topk=1, prefix=None):
    if prefix:
        print('---- Validation results on %s dataset ----' % prefix)

    # Valid set
    val_em, val_f1, val_count = validate_dataset(model, 'dev', tokenizer, device, topk, prefix)
    val_avg_em = 100 * val_em / val_count
    val_avg_f1 = 100 * val_f1 / val_count

    # Test set
    test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, device, topk, prefix)
    test_avg_em = 100 * test_em / test_count
    test_avg_f1 = 100 * test_f1 / test_count
    
    print('%d-best | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
        % (topk, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))
    return val_avg_em


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print('Usage: python3 train_bert.py cuda:<n> <model_path> <save_path>')
        exit(1)


    # Config
    lr = 3e-5
    batch_size = 4
    accumulate_batch_size = 64
    
    assert accumulate_batch_size % batch_size == 0
    update_stepsize = accumulate_batch_size // batch_size
    
    model_path = sys.argv[2]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path)

    device = torch.device(sys.argv[1])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('bert', 'train', tokenizer, batch_size=batch_size, num_workers=16)
    n_step_per_epoch = len(dataloader)
    n_step_per_validation = n_step_per_epoch // 20
    print('%d steps per epoch.' % n_step_per_epoch)
    print('%d steps per validation.' % n_step_per_validation)

    print('Start training...')
    while True:
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

            input_ids = input_ids.cuda(device=device)
            attention_mask = attention_mask.cuda(device=device)
            token_type_ids = token_type_ids.cuda(device=device)
            start_positions = start_positions.cuda(device=device)
            end_positions = end_positions.cuda(device=device)
    
            model.train()
            loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                               start_positions=start_positions, end_positions=end_positions)[0]
            loss.backward()
            step += 1
            print('step %d | Training...\r' % step, end='')   
            if step % update_stepsize == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % n_step_per_validation == 0:
                print("step %d | Validating..." % step)
                val_f1 = validate(model, tokenizer, device, topk=1)
                if val_f1 > best_val:
                    patience = 0
                    best_val = val_f1
                    best_state_dict = deepcopy(model.state_dict())
                    save_path = join(sys.argv[3], 'state_dict.pt')
                    torch.save(best_state_dict, save_path)
                else:
                    patience += 1

            if patience >= 40 or step >= 200000:
                print('Finish training. Scoring 1-5 best results...')
                save_path = join(sys.argv[3], 'state_dict.pt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                for k in range(1, 6):
                    validate(model, tokenizer, device, topk=k)
                del model, dataloader
                exit(0)
