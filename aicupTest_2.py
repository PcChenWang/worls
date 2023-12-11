"""### import package"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np
from tqdm import tqdm, trange
from torch.optim import AdamW

from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import re
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset


def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_torch_seed()

def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()

"""### 資料處理"""

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
ner = '\n\n####\n\n'
special_tokens_dict = {'bos_token': bos,
                       'eos_token': eos,
                       'pad_token': pad,
                       'sep_token': ner}

def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案

    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)

    bounary , item_idx , temp_seq , seq_pairs = 0 , 0 , "" , []
    new_line_idx = 0
    for w_idx, word in enumerate(article):
        if word == '\n':
            new_line_idx = w_idx + 1
            if article[bounary:new_line_idx] == '\n':
                continue
            if temp_seq == "":
                temp_seq = "PHI:Null"
            sentence = article[bounary:new_line_idx].strip().replace('\t' , ' ')
            temp_seq = temp_seq.strip('\\n')
            seq_pair = f"{txt_name}\t{new_line_idx}\t{sentence}\t{temp_seq}\n"
            # seq_pair = special_tokens_dict['bos_token'] + article[bounary:new_line_idx] + special_tokens_dict['sep_token'] + temp_seq + special_tokens_dict['eos_token']
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}:{phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\\n"
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    return seq_pairs

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder , tsv_output_path , num_processes=4):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())

    print("processing each medical file")

    all_seq_pairs = []
    for txt_name in txt_names:
        all_seq_pairs.extend(process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict))
    print(all_seq_pairs[:10])
    print("All medical file done")
    print("write out to tsv format...")
    with open(tsv_output_path , 'w' , encoding = 'utf-8') as fw:
        for seq_pair in all_seq_pairs:
            fw.write(seq_pair)
    print("tsv format dataset done")
    # return all_seq_pairs

anno_info_path = r"C:/Users/qwe86/Desktop/aicup/First_Phase_Release(Correction)/answer.txt"
report_folder = r"C:/Users/qwe86/Desktop/aicup/First_Phase_Release(Correction)/First_Phase_Text_Dataset"
tsv_output_path = './train.tsv'
generate_annotated_medical_report_parallel(anno_info_path, report_folder, tsv_output_path, num_processes=4)

"""### Read Tsv Dataset"""

from datasets import load_dataset, Features, Value

dataset = load_dataset("csv", data_files="./train.tsv", delimiter='\t',
                       features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)

dataset['train'][161]

len(dataset['train'])

"""### Dataloader Sample"""

from transformers import AutoTokenizer, AutoModelForCausalLM

plm = "EleutherAI/pythia-70m" #"EleutherAI/pythia-70m-deduped"

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep ='\n\n####\n\n'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}

tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
tokenizer.padding_side = 'left'
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")

dataset['train']

from torch.utils.data import DataLoader
from islab.aicup import collate_batch_with_prompt_template

train_data = list(dataset['train'])
train_dataloader = DataLoader(train_data, batch_size=3, shuffle=False, collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer))
titer = iter(train_dataloader)
tks, labels, masks= next(titer)
print(tks.shape)
next(iter(titer))

results = tokenizer(
    [f"{bos} 9364819.RAN\\nMINTANIA, JEFFRY {sep} ID: 9364819.RAN\\nNAME: MINTANIA, JEFFRY {eos}",
     f"{bos} This is a sentence {sep} PHI: NULL {eos}"],
    padding=True
)
print(results['attention_mask'][0])
print(results['attention_mask'][1])
print(tokenizer.decode(results['input_ids'][0]))
print(tokenizer.decode(results['input_ids'][1]))

"""### DataLoader For training"""

from islab.aicup import OpenDeidBatchSampler

BATCH_SIZE = 8
bucket_train_dataloader = DataLoader(train_data,
                                     batch_sampler=OpenDeidBatchSampler(train_data, BATCH_SIZE),
                                     collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer),
                                     pin_memory=True)

from transformers import AutoConfig
# the model config to which we add the special tokens
config = AutoConfig.from_pretrained(plm,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)

model = AutoModelForCausalLM.from_pretrained(plm, revision="step3000", config=config)
model

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10 # CHANGE TO THE NUMBER OF EPOCHS YOU WANT
optimizer = AdamW(model.parameters(),lr=3e-5) # YOU CAN ADJUST LEARNING RATE

model.resize_token_embeddings(len(tokenizer))
model.to(device)

from tqdm import tqdm,trange
# 模型儲存資料夾名稱
model_name = "xxx2"
# 模型儲存路徑
model_dir = f"C:/Users/qwe86/Desktop/aicup/models/{model_name}"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
min_loss = 9999

global_step = 0
total_loss = 0

model.train()

train_loss_history = []  # 在訓練之前初始化一個列表，用來存儲每個 epoch 的訓練損失

for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    total_loss = 0

    # Training loop
    predictions , true_labels = [], []

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels, attention_mask=masks)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    
    train_loss_history.append(avg_train_loss)  # 將當前 epoch 的訓練損失加入歷史列表
    
    print("Average train loss: {}".format(avg_train_loss))



    torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_Finial.pt'))
    if avg_train_loss < min_loss:
        min_loss = avg_train_loss
        torch.save(model.state_dict(), os.path.join(model_dir , 'GPT_best.pt'))

 # 繪製損失曲線

plt.plot(range(1 ,EPOCHS + 1),train_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load(os.path.join(model_dir , 'GPT_best.pt')))
model = model.to(device)

def sample_text(model, tokenizer, text, n_words=20):
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None

    with torch.no_grad():
        for _ in range(n_words):
            out = model(inputs, past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            text.append(inputs.item())
            if tokenizer.decode(inputs.item()) == eos:
                break

    return tokenizer.decode(text)

text = special_tokens_dict['bos_token'] + "D.O.B:  29/9/2000" + special_tokens_dict['sep_token']
print(sample_text(model, tokenizer, text=text , n_words=20))

def process_valid_data(test_txts , out_file):
    with open(out_file , 'w' , encoding = 'utf-8') as fw:
        for txt in test_txts:
            m_report = read_file(txt)
            boundary = 0
            # temp = ''.join(m_report)
            fid = txt.split('/')[-1].replace('.txt' , '')
            for idx,sent in enumerate(m_report):
                if sent.replace(' ' , '').replace('\n' , '').replace('\t' , '') != '':
                    sent = sent.replace('\t' , ' ')
                    fw.write(f"{fid}\t{boundary}\t{sent}\n")
                # else:
                #     print(f"{fid}\t{boundary}\t{sent}\n")
                #     assert 1==2
                boundary += len(sent)

test_phase_path = r'C:/Users/qwe86/Desktop/aicup/First_Phase_Release(Correction)/opendid_test'
valid_out_file_path = './valid.tsv'
test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))
test_txts = sorted(test_txts)
valid_data = process_valid_data(test_txts , valid_out_file_path)

from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files=valid_out_file_path, delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])
valid_list

train_phi_category = ['PATIENT', 'DOCTOR', 'USERNAME',
             'PROFESSION',
             'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
             'AGE',
             'DATE', 'TIME', 'DURATION', 'SET',
             'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
             'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']

def get_anno_format(sentence , infos , boundary):
    anno_list = []
    lines = infos.split("\n")
    normalize_keys = ['DATE' , "TIME" , "DURATION" , "SET"]
    phi_dict = {}
    for line in lines:
        parts = line.split(":")
        if parts[0] not in train_phi_category or parts[1] == '':
            continue
        if len(parts) == 2:
            phi_dict[parts[0]] = parts[1].strip()
    for phi_key, phi_value in phi_dict.items():
        normalize_time = None
        if phi_key in normalize_keys:
            if '=>' in phi_value:
                temp_phi_values = phi_value.split('=>')
                phi_value = temp_phi_values[0]
                normalize_time = temp_phi_values[-1]
            else:
                normalize_time = phi_value
        try:
            matches = [(match.start(), match.end()) for match in re.finditer(phi_value, sentence)]
        except:
            continue
        for start, end in matches:
            if start == end:
                continue
            item_dict = {
                        'phi' : phi_key,
                        'st_idx' : start + int(boundary),
                        'ed_idx' : end + int(boundary),
                        'entity' : phi_value,
            }
            if normalize_time is not None:
                item_dict['normalize_time'] = normalize_time
            anno_list.append(item_dict)
    return anno_list

def aicup_predict(model, tokenizer, input, template = "<|endoftext|> __CONTENT__\n\n####\n\n"):
    seeds = [template.replace("__CONTENT__", str(data['content'])) if data['content'] is not None else template for data in input]
    sep = tokenizer.sep_token
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    """Generate text from a trained model."""
    model.eval()
    device = model.device
    texts = tokenizer(seeds, return_tensors = 'pt', padding=True).to(device)
    outputs = []
    #return
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**texts, max_new_tokens=400, pad_token_id = pad_idx,
                                        eos_token_id=tokenizer.convert_tokens_to_ids(eos))
        preds = tokenizer.batch_decode(output_tokens)
        for idx , pred in enumerate(preds):
            if "NULL" in pred:
                continue
            phi_infos = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            annotations = get_anno_format(input[idx]['content'] , phi_infos , input[idx]['idx'])

            for annotation in annotations:
                if 'normalize_time' in annotation:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}\t{annotation["normalize_time"]}')
                else:
                    outputs.append(f'{input[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_idx"]}\t{annotation["ed_idx"]}\t{annotation["entity"]}')
    return outputs

from tqdm.notebook import tqdm
import io
BATCH_SIZE = 32

with open("./answer.txt",'w',encoding='utf8') as f:
    for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
        with torch.no_grad():
            seeds = valid_list[i:i+BATCH_SIZE]
            outputs = aicup_predict(model, tokenizer, input=seeds)
            for o in outputs:
                f.write(o)
                f.write('\n')

file_path = 'C:/Users/qwe86/Desktop/aicup/answer.txt'

# 讀取文件
with open(file_path, 'r') as file:
    # 讀取每一行
    lines = file.readlines()

# 去掉每行開頭的"opendid_test\"
lines = [line.replace('opendid_test\\', '') for line in lines]

# 將處理後的內容寫回文件
with open(file_path, 'w') as file:
    file.writelines(lines)