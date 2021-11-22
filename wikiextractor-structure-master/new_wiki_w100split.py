import csv
import json
import gzip
import logging
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm

logger = logging.getLogger()
import re
import os
from os.path import isfile, join
from fuzzywuzzy import process, fuzz
from multiprocessing import Pool as ProcessPool
from multiprocessing import Process, Manager
from collections import OrderedDict
import unicodedata
import html
import argparse


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def clean_text(text):
    text = _normalize(text)
    text = html.unescape(text)
    return text


def creat_wiki(file_name, pas_id, output_wiki):
    # whole_tsv = OrderedDict()
    whole_tsv = []
    if isfile(file_name):
        f_op = open(file_name, "r")
        file_data = f_op.readlines()
        title_list = []
        title_level = 1
        passage = []
        doc_title = ""
        for line in file_data:
            if re.match("<doc id", line):  ## new document
                if whole_tsv:
                    for section, title_list in whole_tsv:
                        token_all = section.split(" ")
                        token_avg = int(len(token_all) / 100)
                        if float(token_avg) > 0:
                            token_len_avg = int(len(token_all) / float(token_avg))
                            for time_i in range(token_avg):
                                sperate_token = token_all[time_i * token_len_avg:(time_i + 1) * token_len_avg]
                                if len(sperate_token) > 10:
                                    pas_id += 1
                                    text = " ".join(sperate_token)
                                    output_wiki.append(
                                        [pas_id, text, doc_title, " # ".join([tit for tit in title_list])])
                        else:
                            if len(token_all) > 10:
                                pas_id += 1
                                text = " ".join(token_all)
                                output_wiki.append([pas_id, text, doc_title, " # ".join([tit for tit in title_list])])

                title_list = []
                whole_tsv = []
                passage = []
                # whole_tsv = OrderedDict()
            elif re.match(r"<\w\d>.*</\w\d>", line) and line.startswith("<h"):  ## find the subtitle
                if passage != []:
                    passage_txt = " ".join(passage)
                    whole_tsv.append([passage_txt, title_list[0:title_level]])
                sub_title = clean_text(line[4:-6])
                title_level = int(line[2])
                if int(title_level) == 1:
                    doc_title = sub_title
                if title_level > len(title_list):
                    title_list.append(sub_title)
                elif title_level <= len(title_list):
                    title_list[title_level - 1] = sub_title
                passage = []
            elif line == "\n" or line == '\n':
                continue
            else:
                clean_line = re.sub('<.*?>', '', line).replace("&nbsp;[...]", "").replace("&ndash;", "")
                if clean_line == "\n" or clean_line == '\n':
                    continue
                if clean_line != "":
                    if title_list[0:title_level][-1] in ["See also", "Notes", "Footnotes", "References",
                                                         "External links"] or "Further reading" in title_list:
                        continue
                    passage.append(clean_text(clean_line.replace("\n", "")))

        if whole_tsv:
            for section, title_list in whole_tsv:
                token_all = section.split(" ")
                token_avg = int(len(token_all) / 100)
                if float(token_avg) > 0:
                    token_len_avg = int(len(token_all) / float(token_avg))
                    for time_i in range(token_avg):
                        sperate_token = token_all[time_i * token_len_avg:(time_i + 1) * token_len_avg]
                        if len(sperate_token) > 10:
                            pas_id += 1
                            text = " ".join(sperate_token)
                            output_wiki.append(
                                [pas_id, text, doc_title, " # ".join([tit for tit in title_list])])
                else:
                    if len(token_all) > 10:
                        pas_id += 1
                        text = " ".join(token_all)
                        output_wiki.append(
                            [pas_id, text, doc_title, " # ".join([tit for tit in title_list])])

    return pas_id


def write_passages(output, ctx_file):
    logger.info("writing data from: %s", ctx_file)
    with open(ctx_file, "w") as tsvfile:
        writer = csv.writer(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title, title_list
        for doc_output in output:
            row = doc_output
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="the path of the output from the WikiExtractor",
    )
    parser.add_argument(
        "--Output_split_wiki",
        type=str,
        help="list of up-sample rates per each train file. Example: [1,2,1]",
    )
    args = parser.parse_args()
    path = args.path
    all_file = []
    list_dir = sorted(os.listdir(path))
    for list_name in list_dir:
        folder_dict = join(path, list_name)
        file_list = sorted(os.listdir(folder_dict))
        for file_name in file_list:
            all_file.append(join(folder_dict, file_name))

    global output_wiki
    output_wiki = []
    pas_id = 0
    for file_name in tqdm(all_file):
        pas_id = creat_wiki(file_name, pas_id, output_wiki)

    print("the length of wiki passage: ")
    print(len(output_wiki))
    write_passages(output_wiki, args.Output_split_wiki)
