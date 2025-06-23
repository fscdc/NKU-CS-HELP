import glob
import os
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unicodedata
import string

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
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, h, c):
        out, (h, c) = self.rnn(input, (h, c))
        output = self.out(out)
        output = self.softmax(output)
        return output, h, c

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))

n_hidden = 128
rnn = LSTM(n_letters, n_hidden, n_categories).to(device)
criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
    h0, c0 = rnn.initHidden()
    rnn.zero_grad()
    output, h, c = rnn(line_tensor, h0, c0)
    loss = criterion(output[-1], category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
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
    guess, _ = categoryFromOutput(output[-1])
    if guess == category:
        current_acc += 1
    if iter % print_every == 0:
        print('%d %d%% (%s) %.4f %s / %s %s' %
              (iter, iter / n_iters * 100, timeSince(start), loss, line, guess,
               '✓' if guess == category else '✗ (%s)' % category))
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_accs.append(current_acc / plot_every)
        current_loss = 0
        current_acc = 0

plt.figure()
plt.plot(all_losses)
plt.title('Validation Loss')
plt.savefig('torch-lstm_loss.png')

plt.figure()
plt.plot(all_accs)
plt.title('Validation Accuracy')
plt.savefig('torch-lstm_accuracy.png')

confusion = torch.zeros(n_categories, n_categories)

def evaluate(line_tensor):
    h0, c0 = rnn.initHidden()
    output, h, c = rnn(line_tensor, h0, c0)
    return output[-1]

for _ in range(10000):
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
plt.savefig('torch-lstm_confusion.png')
