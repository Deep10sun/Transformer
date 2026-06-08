# Tokenization
## Overview
This document gives a deeper insight into tokenization. It also considers the tradeoff between training custom tokenizer versus using a pre-existing one. 

### Insights
The goal of a tokenizer is to turn raw inputs into numerical data -- essentially creating a mapping --  which can be used for embeddings. Most tokenizers use Byte-Pair Encoding. At a high level this treats each character as a subword, finds most frequent pairs of subwords in the text, then merges them together. This is beneficial for downstream purposes since the one of the goals of this step is to "compress" data.   
eg: "th" shows up a lot in english text ("this" , "that", "there" ... ). Thus, treating "th" as a subword effectively decreases overall number of tokens.

### Tradeoffs
The first tradeoff to consider is whether to create/train a custom tokenizer or use a pre-trained tokenizer. This decision falls on the dataset. If your dataset is unique, it's better to use a custom tokenizer so that your tokens are of higher quality. Pre-trained tokenizers have been trained on countless blogs, articles, newspapers, research papers, etc. Their tokens would be different from custom tokens from a tokenizer trained on the original dataset.  
eg: This project uses the Project Hail Mary's story text as the dataset. The text most likely contains modern slang, general english, but more importantly: Sci-Fi jargon. A custom tokenizer would produce higher quality tokens here with reference to the Sci-Fi jargon as opposed to a pre-trained tokenizer. This would have a direct effect on  the model's accuracy and performance losses downstream. Thus, this repo uses a custom tokenizer.



### Usage
For custom tokenizer training, refer to tokenize.py.  
Then run it:
```bash
python tokenize.py
```
To load the tokenizer, use 
```
tokenizer = Tokenizer.from_file("data/tokenizer.json")
```


