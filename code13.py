import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# TOY DATASET
# -----------------------
sentences = [
    ["I", "love", "NLP"],
    ["She", "enjoys", "learning"],
    ["He", "hates", "math"]
]

tags = [
    ["PRON", "VERB", "NOUN"],
    ["PRON", "VERB", "VERB"],
    ["PRON", "VERB", "NOUN"]
]

# -----------------------
# VOCABULARY
# -----------------------
word2idx = {}
tag2idx = {}

for sent in sentences:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

for tag_seq in tags:
    for tag in tag_seq:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)

# -----------------------
# DATA PREPARATION
# -----------------------
def prepare_sequence(seq, to_idx):
    return torch.tensor([to_idx[w] for w in seq], dtype=torch.long)

training_data = []
for sent, tag_seq in zip(sentences, tags):
    training_data.append((
        prepare_sequence(sent, word2idx),
        prepare_sequence(tag_seq, tag2idx)
    ))

# -----------------------
# UNIDIRECTIONAL LSTM
# -----------------------
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32)
        self.fc = nn.Linear(32, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        scores = self.fc(lstm_out.view(len(x), -1))
        return scores

# -----------------------
# BIDIRECTIONAL LSTM
# -----------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, bidirectional=True)
        self.fc = nn.Linear(32*2, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        scores = self.fc(lstm_out.view(len(x), -1))
        return scores

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_model(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        total_loss = 0

        for sent, tag_seq in training_data:
            model.zero_grad()

            scores = model(sent)

            loss = loss_fn(scores, tag_seq)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model

# -----------------------
# ACCURACY FUNCTION
# -----------------------
def evaluate(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for sent, tag_seq in training_data:
            scores = model(sent)
            _, preds = torch.max(scores, 1)

            correct += (preds == tag_seq).sum().item()
            total += len(tag_seq)

    return correct / total

# -----------------------
# RUN BOTH MODELS
# -----------------------
vocab_size = len(word2idx)
tagset_size = len(tag2idx)

lstm_model = LSTMTagger(vocab_size, tagset_size)
bilstm_model = BiLSTMTagger(vocab_size, tagset_size)

lstm_model = train_model(lstm_model)
bilstm_model = train_model(bilstm_model)

lstm_acc = evaluate(lstm_model)
bilstm_acc = evaluate(bilstm_model)

print("Unidirectional LSTM Accuracy:", lstm_acc)
print("Bidirectional LSTM Accuracy:", bilstm_acc)
