import csv
import json
import unicodedata
import regex as re
from Tokenizer import SimpleTokenizer
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import sys
csv.field_size_limit(sys.maxsize)
def _normalize(text):
    return unicodedata.normalize("NFD", text)


import html
def load_docs_new(ctx_file):
    title_txt = {}
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
                title_txt[title.lower()] = [row[0], text_passage, title_list]
    return title_txt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_doc_path",
    type=str,
    help="Input path of doc-level wikipedia corpus",
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

ctx_file_new = args.input_doc_path
title_text_new = load_docs_new(ctx_file_new)  # subtitle_txt
print("finished loading wikipedia.......")


output = args.gold_data_path
output_intro = args.input_BM25_dataset_path
tok_opts = {}
tokenizer = SimpleTokenizer(**tok_opts)
answer_check = 0
title_check = 0
f = open(output, "r", encoding="utf-8")
with open(output_intro, "r", encoding="utf-8") as f_new:
    data = json.load(f)["data"]  # for NQ dataset
    data_new = json.load(f_new)
    def build_sample_on_whole(idx):
        line = data[idx][0]
        line_new = data_new[idx]
        if idx % 1000 == 0:
            print(idx)
        gold_pass_0_title = line['positive_ctxs'][0]['title']
        BM25_scored_intro = line_new["BM25_scored_ctxs"]
        gold_pass_0_title = _normalize(gold_pass_0_title)
        if gold_pass_0_title in title_text_new:  # check whether title in the new_wikipedia
            gold_doc = title_text_new[gold_pass_0_title]
            new_gold = {"title": gold_pass_0_title, "text": gold_doc[1], "score": 1000, "passage_id": gold_doc[0],  "title_list": gold_doc[2]}
            line_new["positive_ctxs"] = [new_gold]
            hard_negative_intro = []
            for ctxs in BM25_scored_intro:
                retrieval_title = ctxs['title'].lower()
                if retrieval_title != gold_pass_0_title:
                    hard_negative_intro.append({"title": retrieval_title, "text": title_text_new[retrieval_title][1], "score": ctxs['score'],
                        "passage_id": ctxs["passage_id"],  "title_list": ctxs["title_list"]})
            line_new["hard_negative_ctxs"] = hard_negative_intro
            line_new.pop("BM25_scored_ctxs")
            return line_new
        return
    workers_num = 10
    processes = Pool(
        processes=workers_num,
    )
    new_training_data = processes.map(build_sample_on_whole, list(range(len(data_new))))

new_output = []
for line in new_training_data:
    if line and line['positive_ctxs'] and line["hard_negative_ctxs"]:
        new_output.append(line)
print("the length of training_data", len(new_output))
print("Starting saving the data............")
json.dump(new_output, open(args.output_final_dataset_path, "w"))



