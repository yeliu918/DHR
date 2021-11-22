import csv
import json
import gzip
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm

logger = logging.getLogger()
import unicodedata
import random
import argparse
from difflib import SequenceMatcher


##process the psg to doc with title_list
def _normalize(text):
    return unicodedata.normalize("NFD", text)


def load_docs_titlelist(ctx_file):
    title_subtitle = {}
    title_ord_sub = {}
    subtitle_text = {}
    with open(ctx_file) as tsvfile:
        reader = csv.reader(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title, title_list
        for row in reader:
            if row[0] != "id":
                if row[1] == '':
                    print("check")
                if row[2] not in title_ord_sub:
                    title_ord_sub[row[2]] = row[3]
                if row[3] not in subtitle_text:
                    subtitle_text[row[3]] = [row[1]]
                else:
                    subtitle_text[row[3]].append(row[1])

                title_list = row[3].split(" # ")
                curr = title_subtitle
                for cur_title in title_list:
                    cur_title = _normalize(cur_title)
                    if cur_title not in curr:
                        new_node = OrderedDict()
                        curr[cur_title] = new_node
                    curr = curr[cur_title]

    return title_ord_sub, subtitle_text, title_subtitle


def preorder_traverse(root, cur_title_list):
    for cur_title in root:
        cur_title_list.append(cur_title.replace('\'', "").replace('\"', ""))
        preorder_traverse(root[cur_title], cur_title_list)
    return cur_title_list


def write_passages(whole_docs, ctx_file):
    logger.info("writing data from: %s", ctx_file)
    with open(ctx_file, "w") as tsvfile:
        writer = csv.writer(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title, title_list
        for doc in whole_docs:
            writer.writerow(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_psg_path",
        type=str,
        help="Input path of passage wikipedia corpus",
    )
    parser.add_argument(
        "--output_doc_path",
        type=str,
        help="Output path of the document wikipedia corpus",
    )
    args = parser.parse_args()

    title_ord_sub, subtitle_text, title_subtitle = load_docs_titlelist(args.input_psg_path)
    docs = []
    did = 0
    for title in title_ord_sub.keys():
        first_sub = title_ord_sub[title]
        if first_sub == '':
            print('check')
        text = subtitle_text[first_sub]
        title_whole_dict = title_subtitle[title]
        tilte_whole_string = ' # '.join(preorder_traverse(title_whole_dict, []))
        did += 1
        docs.append([did, text, title, tilte_whole_string])
    print("the number of doc is %s .....", len(docs))
    write_passages(docs, args.output_doc_path)
