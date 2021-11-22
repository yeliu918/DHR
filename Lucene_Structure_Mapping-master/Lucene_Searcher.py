import unicodedata
import regex as re
from Tokenizer import SimpleTokenizer
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process, Manager
import sys
import csv
csv.field_size_limit(sys.maxsize)
import argparse


def load_passages(ctx_file: str):
    docs_id = {}
    print("Reading data from: %s", ctx_file)
    with open(ctx_file) as tsvfile:
        reader = csv.reader(
            tsvfile,
            delimiter="\t",
        )
        # file format: doc_id, doc_text, title, title_list
        for row in reader:
            if row[0] != "id":
                if len(row) == 3:
                    docs_id[row[0]] = [row[1], row[2]]
                elif len(row) == 4:
                    docs_id[row[0]] = [row[1], row[2], row[3]]
    return docs_id

def _normalize(text):
    return unicodedata.normalize("NFD", text)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_psg_path",
    type=str,
    help="Input path of passage wikipedia corpus",
)
parser.add_argument(
    "--input_index_path",
    type=str,
    help="Output path of the index",
)
parser.add_argument(
    "--output_BM25_dataset_path",
    type=str,
    help="Output path of the BM25 searched retrieved psg-level training/dev dataset",
)
parser.add_argument(
    "--query_data_path",
    type=str,
    help="original dataset",
)

args = parser.parse_args()
wiki_file = args.input_psg_path
wiki_dump = load_passages(wiki_file)
print("Finished Loading Passages.....")
query_dict = args.query_data_path
output = args.output_training_dataset_path

def process_passage(row):
    import lucene
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field
    from org.apache.lucene.search import IndexSearcher
    from org.apache.lucene.index import DirectoryReader
    from org.apache.lucene.queryparser.classic import QueryParser
    from java.nio.file import Paths
    from org.apache.lucene.store import SimpleFSDirectory
    from org.apache.lucene.search.similarities import BM25Similarity
    lucene.initVM()
    fsDir = SimpleFSDirectory(Paths.get(args.input_index_path))
    reader = DirectoryReader.open(fsDir)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    analyzer = StandardAnalyzer()

    question = row[0]
    answers = row[1]
    Gold_pass = []
    hard_neg_pass = []

    for answer in answers:
        search_q = question + " " + answer
        query = QueryParser("text", analyzer).parse(QueryParser.escape(search_q))
        MAX = 200
        hits = searcher.search(query, MAX)

        # print("Found %d document(s) that matched query '%s':" % (hits.totalHits, query))
        for hit in hits.scoreDocs:
            # print(hit.score, hit.doc, hit.toString())
            doc = searcher.doc(hit.doc)
            doc_text = wiki_dump[doc.get("id")][0]
            doc_title = doc.get("title")
            doc_ids = doc.get("id")
            doc_title_list = wiki_dump[doc.get("id")][2]

            if has_answer(answers, doc_text, tokenizer, "string"):
                if {"title": doc_title, "text": doc_text, "score": hit.score, "passage_id": doc_ids, "title_list": doc_title_list} not in Gold_pass:
                    Gold_pass.append({"title": doc_title, "text": doc_text, "score": hit.score, "passage_id": doc_ids, "title_list": doc_title_list})
            else:
                if {"title": doc_title, "text": doc_text, "score": hit.score, "passage_id": doc_ids, "title_list": doc_title_list} not in hard_neg_pass:
                    hard_neg_pass.append({"title": doc_title, "text": doc_text, "score": hit.score, "passage_id": doc_ids, "title_list": doc_title_list})

    ## add more question related when the hard_neg_pass is small
    if len(hard_neg_pass) < 50:
        query = QueryParser("text", analyzer).parse(QueryParser.escape(question))
        MAX = 200
        hits = searcher.search(query, MAX)
        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)
            doc_text = wiki_dump[doc.get("id")][0]
            doc_title = doc.get("title")
            doc_ids = doc.get("id")
            doc_title_list = wiki_dump[doc.get("id")][2]
            if not has_answer(answers, doc_text, tokenizer, "string"):
                new_hard_neg = {"title": doc_title, "text": doc_text, "score": hit.score, "passage_id": doc_ids, "title_list": doc_title_list}
                if new_hard_neg not in hard_neg_pass:
                    hard_neg_pass.append(new_hard_neg)

    Sorted_Gold_pass = sorted(Gold_pass, key=lambda item: -item['score'])
    Sorted_hard_neg_pass = sorted(hard_neg_pass, key=lambda item: -item['score'])

    return {"dataset": "Natural_Questions", "question": question, "answers": answers[:],
            "positive_ctxs": Sorted_Gold_pass[0:50],  "hard_negative_ctxs": Sorted_hard_neg_pass[0:50]}


tok_opts = {}
tokenizer = SimpleTokenizer(**tok_opts)
data = []

workers_num = 50
processes = Pool(
    processes=workers_num,
)
lines = []
with open(query_dict) as ifile:
    read_lines = csv.reader(ifile, delimiter="\t")
    for line in read_lines:
        if line[1] and eval(line[1]):
            lines.append([line[0], eval(line[1])[0:20]])

data = processes.map(process_passage, lines)

print("Finish Searching")
json.dump(data, open(output, "w"))
print("done!")
