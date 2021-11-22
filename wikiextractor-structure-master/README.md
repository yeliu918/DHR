# Process the Wikipedia page

## Get the clean Wikipedia data with the section title

We treate the wikipeida page as a structural document, which contains a document title, abstract, table of contents and
different levels of sections consisting of titles and paragraphs.

### Download Wikipedia html Dataset

Download the [wikipedia-20181220](https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2)

### Download Wikipedia Extractor

Download the Wikiextractor
https://github.com/attardi/wikiextractor

Delete the "h"s in ignoredTags

```
ignoredTags = [
    'abbr', 'b', 'big', 'blockquote', 'center', 'cite', 'em',
    'font', 'hiero', 'i', 'kbd',
    'p', 'plaintext', 's', 'span', 'strike', 'strong',
    'tt', 'u', 'var'
]
```

Process the Wikiextractor

```
python -m wikiextractor.WikiExtractor enwiki-20181220-pages-articles.xml.bz2  -o output
```

It will produce the struture documents, which contains title in front of each section.

```
[
  {
	"Doc_Title": "....",
	"Abstract": "....",
	"Section 1": [{
		"Section title": "...",
		"text": "..."
	}],
	"hard_negative_ctxs": [“…”],
	“hard_sec_ctxs": ["..."],  # contains this in Psg-level training data
  },
  ...
]
```

## Split the Structured Wikipedia data to the passages

In previous DPR, all the texts under the same document are first concatenated as a single block, which is then split
into multiple blocks of fixed-length passages of 100 words, discarding the blocks of shorter than 100 words. We assume
that the context under the same section contains the same meaning. To avoid ending up with abruptly broken and unnatural
passages, we concatenate the text under the same section and split each section into multiple, disjoint text blocks,
whose maximum length is not over 100 words.

```
python new_wiki_w100split.py \
        --path {output_dir_using_Wikiextractor} \
        --Output_split_wiki "new_psgs_w100.tsv"
```

## Get the structured document

In order to enable the document encoder to capture holistic view of all topics in one document, we use abstract and
table of contents as the document context.

```
python process_doc_w100.py \
    --input_psg_path "new_psgs_w100.tsv" \
    --output_doc_path "docs_w100_title.tsv"
```