import re
import os
import json
import torch
from datetime import datetime
from torch.nn.functional import softmax
from TimeNormalizer import TimeNormalizer
from os.path import join, abspath, dirname
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForQuestionAnswering


class DateDurationQA():
    def __init__(self, tokenizer, model, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
            
        # Other tools
        date_dur_par = ('^(((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|后)(周年|年|岁){0,1}){0,1}'
                '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(周){0,1}){0,1}'
                '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(个){0,1}(月){0,1}){0,1}'
                '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|号|天){0,1}){0,1}$')
        date_dur_full = ('^(((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|后)(年|岁)){0,1}'
                         '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(个){0,1}月){0,1}'
                         '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+周年){0,1}'
                         '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+周){0,1}'
                         '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|号|天)){0,1}$')
        date_pattern = ('(((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|后)年'
                        '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+月){0,1}'
                        '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|号)){0,1}|'
                        '(\d|\.|零|一|二|三|四|五|六|七|八|九|十)+月'
                        '((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|号)){0,1}|'
                        '(\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|号))')
        dur_pattern = ('((\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(周年|年|岁)'
                       '|(\d|\.|零|一|二|三|四|五|六|七|八|九|十)+个月'
                       '|(\d|\.|零|一|二|三|四|五|六|七|八|九|十)+周'
                       '|(\d|\.|零|一|二|三|四|五|六|七|八|九|十)+(日|天))')
        self.date_dur_par_re = re.compile(date_dur_par)
        self.date_dur_full_re = re.compile(date_dur_full)
        self.date_re = re.compile(date_pattern)
        self.dur_re = re.compile(dur_pattern)
        self.lifespan_re = re.compile('\d+年\d+月\d+日－\d+年\d+月\d+日')
        self.birth_re = re.compile('\d+年\d+月\d+日－')
        self.yearspan_re = re.compile('\d{4}-\d{4}')
        self.year_re = re.compile('\d+年')
        self.age_re = re.compile('(\d+(\.\d+)*岁)')
        self.month_day_re = re.compile('^\d+月(\d+(日|号)){0,1}$')
        self.simple_date_re = re.compile('\d+\.\d+\.\d+')
        self.tn = TimeNormalizer()
        self.max_length = 20
        
    def chinese2digits(self, chn):
        chn = chn.replace('两', '二')
        def _trans(s):
            num = 0
            if s:
                idx_q, idx_b, idx_s = s.find('千'), s.find('百'), s.find('十')
                if idx_q != -1:
                    num += digit[s[idx_q - 1:idx_q]] * 1000
                if idx_b != -1:
                    num += digit[s[idx_b - 1:idx_b]] * 100
                if idx_s != -1:
                    num += digit.get(s[idx_s - 1:idx_s], 1) * 10
                if s[-1] in digit:
                    num += digit[s[-1]]
            return num
        digit = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        chn = chn.replace('零', '')
        idx_y, idx_w = chn.rfind('亿'), chn.rfind('万')
        if idx_w < idx_y:
            idx_w = -1
        num_y, num_w = 100000000, 10000
        if idx_y != -1 and idx_w != -1:
            return trans(chn[:idx_y]) * num_y + _trans(chn[idx_y + 1:idx_w]) * num_w + _trans(chn[idx_w + 1:])
        elif idx_y != -1:
            return trans(chn[:idx_y]) * num_y + _trans(chn[idx_y + 1:])
        elif idx_w != -1:
            return _trans(chn[:idx_w]) * num_w + _trans(chn[idx_w + 1:])
        num = _trans(chn)
        if num >= 1000:
            return chn
        else:
            return num
        
    def augment_simple_date(self, passage):
        for match in self.simple_date_re.findall(passage):
            try:
                split = match.split('.')
                if len(split) != 3:
                    continue
                year, month, day = map(int, split)
                if month not in range(1,13) or day not in range(1,32):
                    continue
                new_date = '%d年%d月%d日' % (year, month, day)
                former, latter = passage.split(match)
                passage = '%s%s%s' % (former, new_date, latter)
            except:
                pass
        return passage

    def find_all_range(self, par_re, full_re, tokens):
        assert '' not in tokens
        cursor = 0
        cand_tokens = []
        found_strings = []
        for i, token in enumerate(tokens):
            cand_string = self.tokenizer.convert_tokens_to_string(cand_tokens + [token]).replace(' ', '')
            match = par_re.search(cand_string)
            if not match:
                if cand_tokens:
                    cand_string = self.tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
                    full_match = full_re.search(cand_string)
                    if full_match:
                        found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
                    cand_tokens = []
                    if par_re.match(token):
                        cand_tokens.append(token)
                        cursor = i
                    else:
                        cursor = i + 1
                else:
                    cursor = i + 1
            else:
                cand_tokens.append(token)
        if cand_tokens:
            cand_string = self.tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
            full_match = full_re.search(cand_string)
            if full_match:
                found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
        return found_strings

    def find_sub_list(self, sl, l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll,''.join(sl)))
        return results

    def find_all_date_durs_range(self, p_tokens):
        all_date_durs = self.find_all_range(self.date_dur_par_re, self.date_dur_full_re, p_tokens)
        all_dates, all_durs = [], []
        for ind, ind_end, cand in all_date_durs:
            date_match = self.date_re.search(cand)
            if date_match and date_match.group():
                all_dates.append((ind, ind_end, date_match.group()))
            dur_match = self.dur_re.search(cand)
            if dur_match and dur_match.group():
                all_durs.append((ind, ind_end, dur_match.group()))

        year_bottom = self.find_sub_list(['年', '底'], p_tokens)
        if year_bottom:
            all_dates += year_bottom
        return list(set(all_dates) | set(all_durs))

    def remove_substr(self, datedurs):
        i = 0
        while i < len(datedurs):
            substr = False
            for e in datedurs:
                if datedurs[i][-1] != e[-1] and datedurs[i][-1] in e[-1] and \
                   datedurs[i][0] in range(e[0], e[1]):
                    datedurs.pop(i)
                    substr = True
                    break
            if not substr:
                i += 1
        return datedurs
        
    def tokenize_no_unk(self, text):
        split_tokens = []
        for token in self.tokenizer.basic_tokenizer.tokenize(text, never_split=self.tokenizer.all_special_tokens):
            wp_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            if wp_tokens == [self.tokenizer.unk_token]:
                split_tokens.append(token)
            else:
                split_tokens.extend(wp_tokens)
        return split_tokens
    
    def process_data(self, passage, question, bwd=False):
        # Simplify and tokenize
        p_tokens_no_unk = self.tokenize_no_unk(passage)
        q_tokens_no_unk = self.tokenize_no_unk(question)
        p_tokens = self.tokenizer.tokenize(passage)
        q_tokens = self.tokenizer.tokenize(question)
        assert len(p_tokens_no_unk) == len(p_tokens)
        assert len(q_tokens_no_unk) == len(q_tokens)

        # Truncate input to 512
        overlen = len(p_tokens) + len(q_tokens) - 509
        if overlen > 0:
            if bwd:
                p_tokens_no_unk = p_tokens_no_unk[overlen:]
                p_tokens = p_tokens[overlen:]
            else:
                p_tokens_no_unk = p_tokens_no_unk[:-overlen]
                p_tokens = p_tokens[:-overlen]

        # input = [CLS] (Question) [SEP] (Passage) [SEP]
        input_tokens_no_unk = [self.tokenizer.cls_token, *q_tokens_no_unk, self.tokenizer.sep_token, *p_tokens_no_unk, self.tokenizer.sep_token]
        input_tokens = [self.tokenizer.cls_token, *q_tokens, self.tokenizer.sep_token, *p_tokens, self.tokenizer.sep_token]
        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(input_tokens))
        token_type_ids = torch.LongTensor([0, *(0 for _ in q_tokens), 0, *(1 for _ in p_tokens), 1])
        margin_mask = torch.FloatTensor([-1e10, *(-1e10 for _ in q_tokens), -1e10, *(0. for _ in p_tokens), -1e-10])
        return input_tokens_no_unk, input_ids, token_type_ids, margin_mask
    
    def predict_answer(self, passage, question, bwd=False, topk=5, maxlen=20):
        input_tokens_no_unk, input_ids, token_type_ids, margin_mask = self.process_data(passage, question, bwd=bwd)
        input_ids = input_ids.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        margin_mask = margin_mask.unsqueeze(0)
        if self.device:
            input_ids = input_ids.cuda(self.device)
            token_type_ids = token_type_ids.cuda(self.device)
            margin_mask = margin_mask.cuda(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)    
        
        start_logits, end_logits = outputs[0], outputs[1]
        start_logits += margin_mask
        end_logits += margin_mask
        
        start_probs = softmax(start_logits, dim=1)
        start_probs, start_index = start_probs.topk(topk, dim=1)
        
        probs, preds = [], []
        for n in range(topk):
            start_ind = start_index[0][n].item()
            beam_end_logits = end_logits.clone()
            beam_end_logits[0, :start_ind] += -1e10
            beam_end_logits[0, start_ind+maxlen:] += -1e10
            
            end_probs = softmax(beam_end_logits, dim=1)
            end_probs, end_index = end_probs.topk(1, dim=1)
            end_ind = end_index[0][0]
            
            prob = (start_probs[0][n] * end_probs[0][0]).item()
            span_tokens = input_tokens_no_unk[start_ind:end_ind+1]
            pred = ''.join(self.tokenizer.convert_tokens_to_string(span_tokens).split())
            if pred and pred not in preds:
                probs.append(prob)
                preds.append(pred)
            else:
                probs[preds.index(pred)] += prob
                
        sorted_probs_preds = list(reversed(sorted(zip(probs, preds))))
        probs, preds = map(list, zip(*sorted_probs_preds))
        return probs, preds
    
    def predict_datedur(self, passage, question, bwd=False, topk=5, maxlen=20):
        # Forward prediction
        input_tokens_no_unk, input_ids, token_type_ids, margin_mask = self.process_data(passage, question, bwd=bwd)
        datedurs = self.find_all_date_durs_range(input_tokens_no_unk)
        if len(datedurs) > 0:
            datedurs = self.remove_substr(datedurs)
            datedur_mask = [0 for _ in input_tokens_no_unk]
            for datedur in datedurs:
                start, end, span = datedur
                for cur in range(start, end):
                    datedur_mask[cur] = 1
            datedur_mask = torch.LongTensor(datedur_mask)
            
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            margin_mask = margin_mask.unsqueeze(0)
            datedur_mask = datedur_mask.unsqueeze(0)
            if self.device:
                input_ids = input_ids.cuda(self.device)
                token_type_ids = token_type_ids.cuda(self.device)
                margin_mask = margin_mask.cuda(self.device)
                datedur_mask = datedur_mask.cuda(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=token_type_ids)

            start_logits, end_logits = outputs[0], outputs[1]
            start_logits += margin_mask
            end_logits += margin_mask
            start_probs = softmax(start_logits, dim=1)
            end_probs = softmax(end_logits, dim=1)

            probs, preds = [], []
            for datedur in datedurs:
                start_ind, end_ind, pred = datedur
                prob = (start_probs[0][start_ind] * end_probs[0][end_ind]).item()
                if pred and pred not in preds:
                    probs.append(prob)
                    preds.append(pred)
                else:
                    probs[preds.index(pred)] += prob

            sorted_probs_preds = list(reversed(sorted(zip(probs, preds))))
            probs, preds = map(list, zip(*sorted_probs_preds))
            return probs, preds
        else:
            return [], []
        
    def direct_predict(self, passage, question, topk=3):
        passage = self.augment_simple_date(passage)

        # Regular top-k
        fwd_probs, fwd_preds = self.predict_answer(passage, question, bwd=False, topk=12, maxlen=20)
        bwd_probs, bwd_preds = self.predict_answer(passage, question, bwd=True, topk=12, maxlen=20)
        probs, preds = fwd_probs, fwd_preds
        for prob, pred in zip(bwd_probs, bwd_preds):
            if pred in preds:
                idx = preds.index(pred)
                if prob > probs[idx]:
                    probs[idx] = prob
                #probs[idx] = (probs[idx] + prob) / 2
            else:
                preds.append(pred)
                probs.append(prob)

        # Date-duration entities
        fwd_probs, fwd_preds = self.predict_datedur(passage, question, bwd=False, topk=12, maxlen=20)
        bwd_probs, bwd_preds = self.predict_datedur(passage, question, bwd=True, topk=12, maxlen=20)
        for prob, pred in zip(fwd_probs+bwd_probs, fwd_preds+bwd_preds):
            if pred in preds:
                idx = preds.index(pred)
                if prob > probs[idx]:
                    probs[idx] = prob
            else:
                preds.append(pred)
                probs.append(prob)

        sorted_probs_preds = list(reversed(sorted(zip(probs, preds))))
        probs, preds = map(list, zip(*sorted_probs_preds))

        try:
            ############## POST PROCESSING ##############
            new_preds = []
            new_probs = []
            ############# 生死壽命 (RULE 1) ##############
            birth_year, death_year, age = None, None, None
            lifespans = yearspan_re.findall(passage) # 1970-1980
            if lifespans:
                birth_year, death_year = map(int, lifespans[-1].split('-'))
                age = abs(death_year - birth_year)

            lifespans = lifespan_re.findall(passage)  # 年月日
            if lifespans:
                lifespan = lifespans[0]
                res = json.loads(tn.parse(t2s.convert(lifespan)))
                if 'timespan' in res:
                    birth, death = res['timespan']
                    birth_date = datetime.strptime(birth.split()[0], '%Y-%m-%d')
                    death_date = datetime.strptime(death.split()[0], '%Y-%m-%d')
                    birth_year = birth_date.year
                    death_year = death_date.year
                    age = (death_date - birth_date).days // 365
                    #age = death_year - birth_year

            births = birth_re.findall(passage)
            if births and not lifespans:
                birth = births[0]
                res = json.loads(tn.parse(t2s.convert(birth)))
                if 'timestamp' in res:
                    birth = res['timestamp']
                    birth_date = datetime.strptime(birth.split()[0], '%Y-%m-%d')
                    birth_year = birth_date.year
                    age = (today_date - birth_date).days // 365

            if death_year and age:
                if any(keyword in question for keyword in ('享年', '过世', '过逝', '去世', '冥诞')):
                    if '岁' in question:
                        new_pred = '%d岁' % age
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                    elif '年' in question:
                        new_pred = '%d年' % death_year
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            if birth_year and age:
                if any(keyword in question for keyword in ('出生', '诞生')):
                    if '年' in question:
                        new_pred = '%d年' % birth_year
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                if any(keyword in question for keyword in ('现年', '目前', '现在', '今年')):
                    if '岁' in question:
                        new_pred = '%d岁' % age
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            ############# 年/岁轉換 (RULE 2) #############
            if '岁' in question and '年' not in question or '几岁' in question:
                if not birth_year:
                    pred = preds[0]
                    if pred.endswith('年'):
                        new_pred = '%s岁' % pred[:-1]
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                else:
                    try:
                        pred = preds[0]
                        if '年' in pred:
                            pred_year = int(pred.split('年')[0])
                            diff_year = abs(pred_year - birth_year) - 1
                            new_pred = '%s岁' % diff_year
                            new_preds.append(new_pred)
                            new_probs.append(1.0)
                    except:
                        pass

            if '年' in question and '岁' not in question:
                pred = preds[0]
                if pred.endswith('岁'):
                    new_pred = '%s年' % pred[:-1]
                    new_preds.append(new_pred)
                    new_probs.append(1.0)
            ########## 假設當下时間 (民国/西元) ##########
            pred = preds[0]
            if any(keyword in pred for keyword in ('前天', '昨天', '今天', '明天', '后天', \
                                                   '前年', '去年', '今年', '明年', '后年')):
                res = json.loads(tn.parse(t2s.convert(pred), timeBase=today_date))
                if 'timestamp' in res:
                    pred_date = datetime.strptime(res['timestamp'].split()[0], '%Y-%m-%d')
                    year = pred_date.year
                    month = pred_date.month
                    day = pred_date.day
                    # Almanac conversion
                    if any(keyword in question for keyword in ('民国', '国历')) \
                       and '民国' not in passage:  # P: 西元, Q: 民国
                        if year > 1911:
                            year -= 1911
                    if '西元' in question and '民国' in passage:  # P: 民国, Q: 西元
                        year += 1911
                    # Minimalize answer
                    if any(keyword in pred for keyword in ('日', '号', '天')):
                        new_pred = '%d年%d月%d日' % (year, month, day)
                        if '西元' in question:
                            new_pred = '西元%s' % new_pred
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                    elif '月' in pred:
                        new_pred = '%d年%d月' % (year, month)
                        if '西元' in question:
                            new_pred = '西元%s' % new_pred
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                    elif '年' in pred:
                        new_pred = '%d年' % (year)
                        if '西元' in question:
                            new_pred = '西元%s' % new_pred
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            ################ 假設年分一致 ################
            pred = preds[0]
            year = None
            year_matches = year_re.findall(passage)
            if ('年' in passage or '月' in question) and not year_matches:  # year = 今年
                year = today_date.year
            if len(set(year_matches)) == 1:  # year = 唯一出现的年份
                year = int(year_matches[0][:-1])
            if year:
                if any(keyword in question for keyword in ('民国', '国历')) \
                   and year > 1911:
                    year -= 1911
                match = month_day_re.match(pred)
                if match:
                    if '月' in pred and any(keyword in pred for keyword in ('日', '号')):
                        new_pred = '%d年%s' % (year, pred)
                        if '西元' in question:
                            new_pred = '西元%s' % new_pred
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                    elif '月' in pred:
                        new_pred = '%d年%s' % (year, pred)
                        if '西元' in question:
                            new_pred = '西元%s' % new_pred
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            ############### 假設月份一致 #################
            pred = preds[0]
            if pred.endswith('日') or pred.endswith('号'):
                if pred[:-1] != '' and all(c in '0123456789' for c in pred[:-1]):
                    day = int(pred[:-1])
                    if day in range(1, 32):
                        for pos_pred in preds[1:]:
                            pos_month = pos_pred.split('月')[0]
                            if pos_month != '' and all(c in '0123456789' for c in pos_month):
                                month = int(pos_month)
                                if month in range(1, 13):
                                    new_pred = '%d月%d日' % (month, day)
                                    new_preds.append(new_pred)
                                    new_probs.append(1.0)
            ############## Date - duration ##############
            if all(keyword in question for keyword in ('年', '开始')):
                cands = []
                for pred in preds:
                    matches = year_re.findall(pred)
                    if matches:
                        match = matches[-1]
                    else:
                        matches = age_re.findall(pred)
                        if matches:
                            match = matches[-1][0]
                        else:
                            continue
                    cand = float(match[:-1])
                    if cand % 1 == 0:
                        cand = int(cand)
                    cands.append(cand)
                    if len(cands) == 2:
                        new_year = abs(cands[1] - cands[0])
                        if new_year % 1 == 0:
                            new_year = int(new_year)
                        new_year = str(new_year)
                        new_pred = '%s年' % new_year
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
                        break
            ############### Date - date #################
            try:
                if all(keyword in question for keyword in ('花')):
                    cands = []
                    for pred in preds:
                        matches = year_re.findall(pred)
                        if matches:
                            cands.append(pred)
                        else:
                            continue
                        if len(cands) == 2:
                            if '月' in cands[0] and '月' in cands[1]:
                                split_0 = cands[0].split('年')
                                split_1 = cands[1].split('年')
                                year_0 = int(split_0[0])
                                year_1 = int(split_1[0])
                                month_0 = int(split_0[1].split('月')[0])
                                month_1 = int(split_1[1].split('月')[0])
                                if year_0 > year_1:
                                    if month_0 == month_1:
                                        new_year = year_0 - year_1
                                        new_pred = '%d年' % new_year
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                    elif month_0 > month_1:
                                        new_month = month_0 - month_1
                                        new_year = year_0 - year_1
                                        new_pred = '%d年又%d个月' % (new_year, new_month)
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                    else:
                                        new_month = 12 + month_0 - month_1
                                        new_year = year_0 - year_1 - 1
                                        new_pred = '%d年又%d个月' % (new_year, new_month)
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                else:
                                    if month_0 == month_1:
                                        new_year = year_1 - year_0
                                        new_pred = '%d年' % new_year
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                    elif month_0 < month_1:
                                        new_month = month_1 - month_0
                                        new_year = year_1 - year_0
                                        new_pred = '%d年又%d个月' % (new_year, new_month)
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                    else:
                                        new_month = 12 + month_1 - month_0
                                        new_year = year_1 - year_0 - 1
                                        new_pred = '%d年又%d个月' % (new_year, new_month)
                                        new_preds.append(new_pred)
                                        new_probs.append(1.0)
                                break
                            else:
                                year_1 = int(cands[0].split('年')[0])
                                year_2 = int(cands[1].split('年')[0])
                                new_year = abs(year_2 - year_1)
                                new_pred = '%d年' % new_year
                                new_preds.append(new_pred)
                                new_probs.append(1.0)
                                break
            except:
                pass
            ############## Age arithmetic ###############
            if ('岁' in question and '年' not in question or '几岁' in question) and \
               any(keyword in question for keyword in ('多', '增加', '少')):
                cands = []
                for pred in preds:
                    matches = age_re.findall(pred)
                    if matches:
                        match = matches[-1][0]
                    else:
                        continue

                    if '.' in match:
                        cand = float(match[:-1])
                    else:
                        cand = int(match[:-1])
                    cands.append(cand)

                if len(cands) >= 2:
                    src_cand, tgt_cand = cands[0], cands[1]
                    for cand in cands[1:]:
                        if type(cand) == type(src_cand) and tgt_cand != src_cand:
                            tgt_cand = cand
                            break
                    if src_cand != tgt_cand:
                        new_age = round(abs(tgt_cand - src_cand), 1)
                        if new_age % 1 == 0:
                            new_age = int(new_age)
                        new_pred = '%s岁' % new_age
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            ################### 列舉 ####################
            loc = question.find('哪两')
            if loc > -1:
                n_cand = 2
            else:
                loc = question.find('哪三')
                if loc > -1:
                    n_cand = 3
            if loc > -1:
                unit = question[loc+2:loc+3]
                cands = []
                for pred in preds:
                    if pred.endswith(unit):
                        cands.append(pred)
                    if len(cands) == n_cand:
                        cands.reverse()
                        new_pred = '与'.join(cands)
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            #################### 至今 ###################
            if any(keyword in question for keyword in ('已经', '至今')) and '年' in question:
                for pred in preds:
                    matches = year_re.findall(pred)
                    if matches:
                        src_year = int(matches[-1][:-1])
                        tgt_year = today_date.year
                        if '民国' in passage:
                            tgt_year -= 1911
                        new_year = abs(tgt_year - src_year)
                        new_pred = '%d年' % new_year
                        new_preds.append(new_pred)
                        new_probs.append(1.0)
            ############### 補上缺失的單位 ###############
            if not new_preds:
                if '年' in question and '岁' not in question:
                    for i, pred in enumerate(preds):
                        if pred.endswith('年'):
                            if i > 0:
                                new_preds.append(pred)
                                new_probs.append(1.0)
                            break
                        if pred != '' and all(c in '0123456789' for c in pred):
                            new_preds.append(pred + '年')
                            new_probs.append(1.0)
                            break
                elif '岁' in question and '年' not in question or '几岁' in question:
                    for i, pred in enumerate(preds):
                        if pred.endswith('岁'):
                            if i > 0:
                                new_preds.append(pred)
                                new_probs.append(1.0)
                            break
                        if pred != '' and all(c in '0123456789' for c in pred):
                            new_preds.append(pred + '岁')
                            new_probs.append(1.0)
                            break
            ################ 朝代補充 ###################
            if '朝代' in question:
                for pred in preds:
                    for dynasty in ('戰国', '三国'):
                        if dynasty in pred:
                            new_preds.append(dynasty + '时代')
                            new_probs.append(1.0)
            ############## 国立年历轉換 ##################
            if any(keyword in question for keyword in ('民国', '国历')):
                for i in range(len(preds)):
                    if '民国' not in preds[i] and preds[i].endswith('年'):
                        try:
                            year = int(preds[i].split('年')[0])
                            if year > 1911:
                                new_pred = '民国%d年' % (year - 1911)
                                new_preds.append(new_pred)
                                new_probs.append(1.0)
                                break
                        except:
                            pass
            ############### 民国單位補正 #################
            if any(keyword in question for keyword in ('民国', '国历')):
                found, candidate = False, None
                for pred in new_preds + preds:
                    if pred.endswith('年'):
                        try:
                            _ = int(pred.split('年')[0])
                            if not candidate:
                                candidate = pred
                        except:
                            pass
                    if pred.startswith('民国'):
                        new_preds.insert(0, pred)
                        new_probs.insert(0, 1.0)
                        found = True
                        break

                if not found and candidate:
                    new_preds.insert(0, '民国%s' % candidate)
                    new_probs.insert(0, 1.0)
            #############################################
            preds = new_preds + preds
            ############## POST PROCESSING ##############
        except:
            pass
        if len(preds) == 0:
            preds = ['']
            probs = [0.0]
        for i in reversed(range(len(preds))):
            pred = preds[i].strip()
            if not pred or preds[i] == '[SEP]':
                preds.pop(i)
                probs.pop(i)
        preds = preds[:topk]
        probs = probs[:topk]
        return preds, probs

    def predict_single(self, passage, question, topk=3):
        preds, probs = self.direct_predict(passage, question, topk=topk)
        acands = []
        for pred, prob in zip(preds, probs):
            acand = {'AMODULE': 'Date-Duration'}
            acand['ATEXT'] = pred
            acand['score'] = prob
            acand['start_score'] = 0.0
            acand['end_score'] = 0.0
            acands.append(acand)
        return acands
    
    def predict(self, input_data, topk=3):
        acands = []
        passage = input_data['DTEXT']
        all_Q = input_data['QUESTIONS']
        for Q in all_Q:
            question = Q['QTEXT']
            acand = self.predict_single(passage, question, topk)
            acands.append(acand)
        return acands
