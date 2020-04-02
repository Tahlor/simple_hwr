
### Dataset Issues / Considerations

The IAM dataset can be parsed in a number of ways. It is unclear in many publications it is not clear how it was parsed. Hopefully this will at least spell out the variety of possible ways it can be parsed. This repo includes code for processing both lines and words, but not for the entire page.

The images can be either extracted from the the xml and the full page image or directly from the lines.tgz or words.tgz. You will notice in the lines.tgz, there are "bounding boxes" around the words which theoretically could give additional context for a neural network to learn. Ideal the images should be directly cropped from the full page to avoid these artifacts, but we do not include that in this repo as of yet.

There are two main sources for the text ground truth: the xmls files or the txt file (lines.txt or words.txt). We provide code for parsing both. For word recognition they parse to be identical. However on lines there are some differences. In the txt the punctuation is often separated by a space, where in the xml it is not.

`|` represents the space character  
txt: `"|He|is|not|going|to|agree|to|be|bound|over|.|That|will`  
xml: `"He|is|not|going|to|agree|to|be|bound|over.|That|will`

However there are some cases (we probably have not found them all) where punctuation is not delimited. For example, in numbers:

txt: `speech|scared|1,157|East|Germans|to`  
or   
txt: `South|Africa's|exclusion|from|the`  
or  
txt: `Nobody|'s|going|to|shove|2ole|Uncle|Sam|around|!|"|He|then`

Parsing from the xml would probably be the most "fair" method because it is probably closest to what the authors were given as prompts, but most code using the IAM I have seen uses the lines.txt.

Depending on the processing of this text will affect the WER and CER of the final results. If punctuation is considered its own word, that may significantly improve WER because often a system will accurately predict punctuation.

### Download IAM data
This requires a IAM account. To download all of the data that will be need to run this code run the following script.


`sh download_IAM_data.sh`


### Extracting Ground Truth

We provide two ways to extract the ground truth. They are saved in `raw_gts` in word/line and xml/txt directories. Each set is saved in the format:

```
[
    {
        "gt": string,
        "image_path": string,
        "err": bool
    },
    {
        "gt": string,
        "image_path": string,
        "err": bool
    },
    ...
]
```

This code extracts all word and lines:  
`python extract_all_words_lines.py`

### Additional Processing

Additional process may be done on the dataset. This can be done to remove lines marked with errors, remove images are are problematic, compute author specific statistics, etc.

For now we just run this processing
```
sh add_author_std.sh
```

### TLDR

Run

```
sh download_IAM_data.sh
python extract_all_words_lines.py
sh add_author_std.sh
```
