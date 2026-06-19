from tokenizers import Tokenizer
from model.model import LanguageModel
import torch

#Read in raw tokenizer_data
with open("dataset/input.txt", "r", encoding="utf-8") as f:
    story = f.read()

#Tokenize raw tokenizer_data
tokenizer = Tokenizer.from_file("tokenizer_data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
tokens = tokenizer.encode(story)
data = torch.tensor(tokens.ids)
data_len = len(data)

#Split tokenizer_data into train/val/test sets (80-10-10 split)
p_train = 0.8
p_val = p_test = (1 - p_train)/2

train_idx_end = int(data_len*p_train)
val_idx_end = train_idx_end + int(data_len*p_val)

train_set = data[:train_idx_end]
val_set = data[train_idx_end:val_idx_end]
test_set = data[val_idx_end:]

#Create labels (offset by 1) and split them accordingly
labels = data[1:]
label_train = labels[:train_idx_end]
label_val = labels[train_idx_end:val_idx_end]
label_test = labels[val_idx_end:]

#HYPERPARAMETERS
training_steps = 1000
num_epochs = 75
block_size = 128
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lm = LanguageModel(vocab_size) #logits: (Batch, Block, Channels)

#MAIN TRAINING LOOP
for epoch in range(num_epochs):
    print(f"epoch: {epoch+1}")
    for t_step in range(training_steps):
        print(f" {t_step+1}", end=" ")

        #create batch_set
        index = torch.randint(
            0,
            len(train_set) - block_size,
            (batch_size,)
        )
        x = train_set[index[:, None] + torch.arange(block_size)]

        output = lm(x)
        print(output.shape)
        #Feed tokenizer_data into model
        print(f"val_loss: {0}")