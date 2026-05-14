import os
import json
import string
import numpy as np
import argparse
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch
import collections
import csv
import math
import openai
from scipy import stats
from sklearn import metrics
from collections import OrderedDict
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from eval_metrics import evaluate_metrics
from tqdm import tqdm
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

openai.api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API key:", "loaded" if openai.api_key else "NOT SET (set OPENAI_API_KEY env var)")
parser = argparse.ArgumentParser(description='AudioMosaic')
# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--task', type=str, default='captioning')
parser.add_argument('--text_embed_setting', type=str, default='gpt')
parser.add_argument('--eval_filename', default='ltu_inference_outputs/checkpoints_ltu_ori_paper.bin_clotho_results.json', type=str)
parser.add_argument('--label_csv', type=str, default='dataset/ltu_eval_class_labels/class_labels_indices_as.csv')
parser.add_argument('--mode', type=str, default='accu')

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000     # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def get_gpt_embedding(input_text, mdl_size='text-embedding-ada-002'):
    """Get GPT embeddings for single text or batch of texts with proper batching and stacking.
    
    Args:
        input_text: str or list of str
        mdl_size: model name
        
    Returns:
        Single embedding (list) if input_text is str, or stacked embeddings (np.ndarray) if input_text is list
    """
    # Handle both single string and batch of strings
    if isinstance(input_text, str):
        single_input = True
        input_list = [input_text]
    else:
        single_input = False
        input_list = list(input_text)
    
    print(f"Getting GPT embeddings for {len(input_list)} text(s)")
    
    response = embedding_with_backoff(
        input=input_list,
        model=mdl_size
    )
    
    # Extract embeddings and stack them
    embeddings_list = [item['embedding'] for item in response['data']]
    embeddings_array = np.array(embeddings_list)  # Stack as numpy array
    
    if single_input:
        return embeddings_list[0]  # Return single embedding as list
    else:
        return embeddings_array  # Return stacked embeddings as numpy array


def get_bert_embedding(bert_model, bert_tokenizer, input_text):
    input_text = remove_punctuation_and_lowercase(input_text)
    #print(input_text)
    inputs = bert_tokenizer(input_text, return_tensors="pt")
    if inputs['input_ids'].shape[1] > 512:
        inputs['input_ids'] = inputs['input_ids'][:, :512]
        inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
        inputs['attention_mask'] = inputs['attention_mask'][:, :512]
    outputs = bert_model(**inputs.to(device))
    last_hidden_states = torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()
    return last_hidden_states.tolist()


def cosine_similarity(vector1, vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
    magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))
    return dot_product / (magnitude1 * magnitude2)


def chunked_list(items, chunk_size):
    for idx in range(0, len(items), chunk_size):
        yield items[idx: idx + chunk_size]


def make_name_dict(label_csv):
    if "as" in label_csv:
        name_lookup = collections.OrderedDict()
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            line_count = 0
            for row in csv_reader:
                display_name = row['display_name']
                name_lookup[row['mid']] = display_name
                line_count += 1
        return name_lookup
    elif "esc" in label_csv or "bj" in label_csv or "tut" in label_csv:
        label_list = np.loadtxt(label_csv, dtype=str, delimiter=',', skiprows=1)
        label_dict = OrderedDict()
        for i in range(label_list.shape[0]):
            class_code = label_list[i, 1]
            class_name = label_list[i, 2][1:-1]
            label_dict[class_code] = class_name
        return label_dict
    elif "vgg" in label_csv:
        label_list = np.loadtxt(label_csv, dtype=str, delimiter=',', skiprows=1)
        label_dict = OrderedDict()
        for i in range(label_list.shape[0]):
            class_code = label_list[i, 1]
            class_name = label_list[i, 2].replace('_', ', ')
            label_dict[class_code] = class_name
        return label_dict
   


def remove_punctuation_and_lowercase(text):
    """
    This function takes a string as input, removes all the punctuations,
    and converts the string to lowercase.
    """
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text


def main(args):
    eval_file_header = args.eval_filename.replace('.json', '')
    os.makedirs(eval_file_header, exist_ok=True)

    if args.task == 'captioning':
        with open(args.eval_filename, 'r') as f:
            eval_data = json.load(f)
        num_sample = len(eval_data)
        
        pred_dict = {}
        truth_dict = {}
        for i in range(num_sample):
            cur_audio_id = eval_data[i]['audio_id'].split('/')[-1]
            if args.task == 'captioning':
                cur_pred = remove_punctuation_and_lowercase(eval_data[i]['pred'].split(':')[-1][1:])
                cur_truth = remove_punctuation_and_lowercase(eval_data[i]['ref'].split(':')[-1][1:])
            if cur_audio_id in pred_dict.keys():
                pred_dict[cur_audio_id].append(cur_pred)
                truth_dict[cur_audio_id].append(cur_truth)
            else:
                pred_dict[cur_audio_id] = [cur_pred]
                truth_dict[cur_audio_id] = [cur_truth]
        
        if os.path.exists(eval_file_header) == False:
            os.mkdir(eval_file_header)

        ciders, spices = [], []
        result_summary = []
        for trial in range(5):
            all_pred = [['file_name', 'caption_predicted']]
            all_truth = [['file_name', 'caption_reference_01', 'caption_reference_02', 'caption_reference_03', 'caption_reference_04', 'caption_reference_05']]
            for key in pred_dict.keys():
                cur_audio_id = key
                cur_pred = pred_dict[key][trial]
                cur_truth = truth_dict[key]
                all_pred.append([cur_audio_id, cur_pred])
                all_truth.append([cur_audio_id] + cur_truth)

            print(len(all_pred), len(all_truth))
            np.savetxt(eval_file_header + '/' + 'pred_{:d}.csv'.format(trial), all_pred, fmt='%s', delimiter=',')
            np.savetxt(eval_file_header + '/' + 'truth_{:d}.csv'.format(trial), all_truth, fmt='%s', delimiter=',')
        
            res = evaluate_metrics(eval_file_header + '/' + 'pred_{:d}.csv'.format(trial), 
                                eval_file_header + '/' + 'truth_{:d}.csv'.format(trial), 5)
            ciders.append(res['cider']['score'])
            spices.append(res['spice']['score'])

            with open(eval_file_header + "/" + "res_summary_{:d}.json".format(trial), "w") as f:
                json.dump(res, f)
                
        result_summary.append([eval_file_header, np.mean(ciders), np.mean(spices)])
        np.savetxt(eval_file_header + '/' + 'all_clotho_summary_ablation.csv', result_summary, delimiter=',', fmt='%s')
        ciders = ciders + [np.mean(ciders), np.std(ciders)]
        spices = spices + [np.mean(spices), np.std(spices)]

        np.savetxt(eval_file_header + "/ciders_summary.csv", ciders, delimiter=',')
        np.savetxt(eval_file_header + "/spices_summary.csv", spices, delimiter=',')
    elif args.task == 'cls':
        if args.text_embed_setting == 'bert':
            bert_mdl_size = 'bert-large-uncased'
            bert_tokenizer = AutoTokenizer.from_pretrained(bert_mdl_size, model_max_length=512)
            bert_model = BertModel.from_pretrained(bert_mdl_size).to(device)
            
        label_csv = args.label_csv
        ori_label_dict = make_name_dict(label_csv)
        print(ori_label_dict)
        # load cached label embedding dict
        if os.path.exists('{}/label_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting)):
            with open('{}/label_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting), 'r') as f:
                json_str = f.read()
            label_dict = json.loads(json_str, object_pairs_hook=OrderedDict)
            # Convert lists back to numpy arrays
            for key in label_dict:
                if isinstance(label_dict[key], list):
                    label_dict[key] = np.array(label_dict[key])
        else:
            label_dict = OrderedDict()
            class_names = list(ori_label_dict.values())
            
            if args.text_embed_setting == 'gpt':
                # Batch process all labels at once with proper stacking
                label_texts = ['sound of ' + name.lower() for name in class_names]
                embeddings_array = get_gpt_embedding(label_texts)  # Returns np.ndarray [N, embedding_dim]
                for class_name, embedding in zip(class_names, embeddings_array):
                    label_dict[class_name] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            elif args.text_embed_setting == 'bert':
                # Process BERT embeddings individually (BERT doesn't batch as efficiently)
                for class_name in class_names:
                    embedding = get_bert_embedding(bert_model, bert_tokenizer, 'sound of ' + class_name.lower())
                    label_dict[class_name] = embedding if isinstance(embedding, list) else embedding.tolist()

            with open('{}/label_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting), 'w') as f:
                json_str = json.dumps(label_dict)
                f.write(json_str)
        if os.path.exists('{}/embed_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting)) == True:
            with open('{}/embed_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting), 'r') as f:
                embed_cache = f.read()
            embed_cache = json.loads(embed_cache)
            # Convert lists back to numpy arrays
            for key in embed_cache:
                if isinstance(embed_cache[key], list):
                    embed_cache[key] = np.array(embed_cache[key])
        else:
            embed_cache = {}

        with open(args.eval_filename, 'r') as fp:
            eval_data = json.load(fp)

        num_class = len(label_dict)
        def get_pred(cur_pred_list, label_dict):
            # at beginning, all zero scores
            score = np.zeros(num_class)
            label_embed_list = list(label_dict.values())
            
            # Collect predictions that need embedding
            preds_needing_embedding = [p for p in cur_pred_list if p not in embed_cache]
            
            # Batch get embeddings for new predictions
            if preds_needing_embedding:
                if args.text_embed_setting == 'gpt':
                    if len(preds_needing_embedding) == 1:
                        new_embedding = get_gpt_embedding(preds_needing_embedding[0])  # Single string returns list
                        embed_cache[preds_needing_embedding[0]] = new_embedding
                    else:
                        embeddings_array = get_gpt_embedding(preds_needing_embedding)  # Batch returns np.ndarray
                        for pred, embedding in zip(preds_needing_embedding, embeddings_array):
                            embed_cache[pred] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                else:
                    # BERT one at a time
                    for pred in preds_needing_embedding:
                        embedding = get_bert_embedding(bert_model, bert_tokenizer, pred)
                        embed_cache[pred] = embedding
            
            # Now compute scores from cache
            for cur_pred in cur_pred_list:
                cur_pred_embed = embed_cache[cur_pred]
                for i in range(num_class):
                    if args.mode == 'accu':
                        score[i] = score[i] + cosine_similarity(cur_pred_embed, label_embed_list[i])
                    elif args.mode == 'max':
                        score[i] = max(score[i], cosine_similarity(cur_pred_embed, label_embed_list[i]))
            return score

        num_sample = len(eval_data)
        print('number of samples {:d}'.format(num_sample))
        def parse_pred_text(sample):
            pred_text = sample['pred'].replace('"', '').split('Audio caption')[-1][2:]
            return 'sound of ' + pred_text.lower()

        all_pred_texts = [parse_pred_text(sample) for sample in eval_data]
        unique_pred_texts = [p for p in dict.fromkeys(all_pred_texts) if p not in embed_cache]

        if unique_pred_texts:
            if args.text_embed_setting == 'gpt':
                # Batch GPT embedding requests to reduce API calls.
                for batch in chunked_list(unique_pred_texts, 128):
                    batch_embeddings = get_gpt_embedding(batch)
                    if isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = batch_embeddings.tolist()
                    for pred_text, embedding in zip(batch, batch_embeddings):
                        embed_cache[pred_text] = embedding
            else:
                for pred_text in unique_pred_texts:
                    embed_cache[pred_text] = get_bert_embedding(bert_model, bert_tokenizer, pred_text)

        all_pred = np.zeros([num_sample, num_class])
        all_truth = np.zeros([num_sample, num_class])
        for i in tqdm(range(num_sample)):
            cur_audio_id = eval_data[i]['audio_id']
            cur_pred_list = [all_pred_texts[i]]

            cur_truth_list = eval_data[i]['ref'].split(': ')[-1].split('; ')
            cur_truth_list = [item.lower() for item in cur_truth_list]
            print(cur_truth_list)
            for cur_truth in cur_truth_list:
                if cur_truth not in label_dict:
                    if cur_truth.replace("_", " ") in label_dict:
                        cur_truth = cur_truth.replace("_", " ")
                    elif cur_truth.replace("/", " ") in label_dict:
                        cur_truth = cur_truth.replace("/", " ")
                    else:
                        raise(ValueError('warning: ground truth {:s} not in label dict'.format(cur_truth)))
                cur_truth_idx = list(label_dict.keys()).index(cur_truth)
                all_truth[i, cur_truth_idx] = 1.0

            all_pred[i] = get_pred(cur_pred_list, label_dict)
        
        save_fold = "{}/{:s}_cla_report".format(eval_file_header, args.text_embed_setting)
        if os.path.exists(save_fold) == False:
            os.makedirs(save_fold)

        np.save("{}/all_pred.npy".format(eval_file_header), all_pred)
        np.save("{}/all_truth.npy".format(eval_file_header), all_truth)
        stats = calculate_stats(all_pred, all_truth)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        np.savetxt("{}/result_summary.csv".format(eval_file_header), [mAP, mAUC, acc], delimiter=',')

        # Convert numpy arrays to lists for JSON serialization
        embed_cache_serializable = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in embed_cache.items()}
        embed_cache_json = json.dumps(embed_cache_serializable)
        save_cache_path = '{}/embed_cache_{:s}.json'.format(eval_file_header, args.text_embed_setting)
        with open(save_cache_path, 'w') as f:
            f.write(embed_cache_json)

        print('mAP: ', mAP, "mAUC: ", mAUC, "acc: ", acc)

    return 


if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    main(args)