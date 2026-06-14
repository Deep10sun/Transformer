from tokenizers import Tokenizer

#Read in raw data
with open("input.txt", "r", encoding="utf-8") as f:
    story = f.read()

#Tokenize raw data
tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
tokens = tokenizer.encode(story)
data = tokens.ids

#Split data into train/val/test sets (80-10-10 split)
p_train = 0.8
p_val = p_test = (1 - p_train)/2

train_idx_end = int(len(data)*p_train)
val_idx_end = train_idx_end + int(len(data)*p_val)

train_set = data[:train_idx_end]
val_set = data[train_idx_end:val_idx_end]
test_set = data[val_idx_end:]

#Create labels (offset by 1) and split them accordingly
labels = data[1:]
label_train = labels[:train_idx_end]
label_val = labels[train_idx_end:val_idx_end]
label_test = labels[val_idx_end:]