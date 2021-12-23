import csv
import json
from Has_answer import _normalize, has_answer
from fuzzywuzzy import process
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import sys
sys.path.append("..")
from dhr.utils.tokenizers import SimpleTokenizer

csv.field_size_limit(sys.maxsize)
import html
from rank_bm25 import BM25Okapi


def load_psgs_new(ctx_file):
    title_txt = {}
    subtitle_txt = {}
    with open(ctx_file) as tsvfile:
        reader = csv.reader(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title, title_list
        for row in reader:
            if row[0] != "id":
                title = row[2]
                title = _normalize(title)
                title = html.unescape(title)
                title_list = row[3]
                text_passage = html.unescape(row[1])
                if title.lower() not in title_txt:
                    title_txt[title.lower()] = [[row[0], text_passage, row[3]]]
                else:
                    title_txt[title.lower()].append([row[0], text_passage, row[3]])
                if title_list.lower() not in subtitle_txt:
                    subtitle_txt[title_list.lower()] = [[row[0], text_passage, row[2]]]
                else:
                    subtitle_txt[title_list.lower()].append([row[0], text_passage, row[2]])
    return title_txt, subtitle_txt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_psg_path",
    type=str,
    help="Input path of psg-level wikipedia corpus",
)
parser.add_argument(
    "--input_BM25_dataset_path",
    type=str,
    help="Input path of the BM25 searched retrieved doc-level training/dev dataset",
)
parser.add_argument(
    "--gold_data_path",
    type=str,
    help="original dataset with gold answers and passage",
)
parser.add_argument(
    "--output_final_dataset_path",
    type=str,
    help="the final output path of training/dev dataset",
)
args = parser.parse_args()

ctx_file_new = args.input_psg_path
title_text_new, subtitle_txt = load_psgs_new(ctx_file_new)  # subtitle_txt
print("finished loading wikipedia.......")
output = args.gold_data_path
output_new = args.input_BM25_dataset_path

tok_opts = {}
tokenizer = SimpleTokenizer(**tok_opts)
answer_check = 0
title_check = 0
with open(output, "r", encoding="utf-8") as f:
    with open(output_new, "r", encoding="utf-8") as f_new:
        data = json.load(f)["data"]
        data_new = json.load(f_new)


        def build_sample_on_whole(idx):
            line = data[idx][0]
            line_new = data_new[idx]
            if idx % 1000 == 0:
                print(idx)
            if line["question"] == line_new["question"]:
                answers = line["short_answers"]
                gold_pass_0_title = line['positive_ctxs'][0]['title']
                gold_pass_0_title = _normalize(gold_pass_0_title)
                if gold_pass_0_title in title_text_new:  # check whether title in the new_wikipedia
                    text_pool_all = [tuples for tuples in title_text_new[gold_pass_0_title] if
                                     has_answer(answers, tuples[1], tokenizer, "string")]
                    text_poo_exp = [tuples for tuples in title_text_new[gold_pass_0_title] if
                                    not has_answer(answers, tuples[1], tokenizer, "string")]
                    text_pool = [tuples[1] for tuples in text_pool_all]
                    ## get the gold passage
                    new_gold = []
                    if line_new['positive_ctxs'] and line_new['positive_ctxs'][0]['title'] == gold_pass_0_title:
                        new_gold = line_new['positive_ctxs'][0]
                    else:
                        if len(text_pool) > 0:
                            tokenized_corpus = [doc[1].split(" ") for doc in text_pool]
                            bm25 = BM25Okapi(tokenized_corpus)
                            tokenized_query = line["question"].split(" ")
                            passage = bm25.get_top_n(tokenized_query, text_pool_all, n=1)[0]

                            if has_answer(answers, passage[1], tokenizer, "string"):
                                new_gold = {"title": gold_pass_0_title, "text": passage[1], "score": 1000,
                                            "passage_id": passage[0],
                                            "title_list": passage[2]}  # , "context_512": context_passage
                    ## get the in-article negative passage
                    hard_sec_ctxs = []
                    if len(text_poo_exp) > 0:
                        tokenized_corpus = [doc[1].split(" ") for doc in text_poo_exp]
                        bm25 = BM25Okapi(tokenized_corpus)
                        tokenized_query = line["question"].split(" ")
                        bm25_text_poo_exp = bm25.get_top_n(tokenized_query, text_poo_exp, n=100)

                        for neg in bm25_text_poo_exp:
                            if neg[1] not in text_pool:
                                hard_sec_ctxs.append(
                                    {"title": gold_pass_0_title, "text": neg[1], "score": 1000,
                                     "passage_id": neg[0], "title_list": neg[2]})

                    if not hard_sec_ctxs:
                        hard_sec_ctxs = line_new['hard_negative_ctxs'][0:10]

                    line_new["hard_sec_ctxs"] = hard_sec_ctxs

                    if not line_new["positive_ctxs"]:
                        if new_gold:
                            line_new["positive_ctxs"] = [new_gold]
                        else:
                            line_new["positive_ctxs"] = []
                    else:
                        if new_gold:
                            line_new["positive_ctxs"][0] = new_gold
                    return line_new
                else:
                    return
                # return
            return


        workers_num = 50
        processes = Pool(
            processes=workers_num,
        )
        new_training_data = processes.map(build_sample_on_whole, list(range(len(data))))

    new_output = []
    for line in new_training_data:
        if line and line['positive_ctxs'] and line["hard_negative_ctxs"] and "hard_sec_ctxs" in line:
            new_output.append(line)
    print("the length of training_data", len(new_output))
    print("Starting saving the data............")
    json.dump(new_output, open(args.output_final_dataset_path, "w"))
