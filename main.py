from tokenizers import Tokenizer

with open("input.txt", "r", encoding="utf-8") as f:
    story = f.read()

tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
tokens = tokenizer.encode(story)
data = tokens.ids

