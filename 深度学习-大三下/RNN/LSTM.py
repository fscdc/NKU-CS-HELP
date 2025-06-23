import glob
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unicodedata
import string
import torch
import time
import math
import random

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('./data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor.to(device)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long).to(device)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_class=10):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, output_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input_length = input.size()[0]
        hidden = torch.zeros(1, self.hidden_dim, device=device)
        cell = torch.zeros(1, self.hidden_dim, device=device)
        for i in range(input_length):
            x = input[i].view(1, -1)
            state = torch.cat((x, hidden), dim=-1)
            f = torch.sigmoid(self.forget_gate(state))
            i = torch.sigmoid(self.input_gate(state))
            c = torch.tanh(self.cell_update(state))
            cell = f * cell + i * c
            output = torch.sigmoid(self.output_gate(state))
            hidden = output * torch.tanh(cell)
        output = self.softmax(self.classify(hidden))
        return output

n_hidden = 128
lstm = LSTM(n_letters, n_hidden, output_class=n_categories).to(device)
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.005)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    optimizer.zero_grad()
    output = lstm(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

n_iters = 300000
print_every = 5000
plot_every = 1000
current_loss = 0
current_acc = 0
all_losses = []
all_accs = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    guess, _ = categoryFromOutput(output)
    if guess == category:
        current_acc += 1
    if iter % print_every == 0:
        print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess,
            '✓' if guess == category else '✗ (%s)' % category))
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_accs.append(current_acc / plot_every)
        current_loss = 0
        current_acc = 0

plt.figure()
plt.plot(all_losses)
plt.title('Validation Loss')
plt.savefig('lstm_loss.png')

plt.figure()
plt.plot(all_accs)
plt.title('Validation Accuracy')
plt.savefig('lstm_accuracy.png')

confusion = torch.zeros(n_categories, n_categories)

def evaluate(line_tensor):
    return lstm(line_tensor)

for i in range(10000):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('lstm_confusion.png')
