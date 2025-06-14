# -*- coding: UTF-8 -*-
"""
@Project ：SA 
@File ：main.py
@Author ：AnthonyZ
@Date ：2024/10/9 13:26
"""
from model import CNN, LSTM
from utils import *
from data import *

import argparse
import collections
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchtext

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


if __name__ == '__main__':
    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

    print(train_data)
    print(test_data)

    tokenizer = basic_english_normalize
    # TODO：超参数，可自行调整
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", default=256)
    parser.add_argument("--test_size", default=0.25)
    parser.add_argument("--min_freq", default=5)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--embedding_dim", default=300)
    parser.add_argument("--n_filters", default=100)
    parser.add_argument("--filter_sizes", default=[3, 5, 7])
    parser.add_argument("--dropout_rate", default=0.25)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--model", default="lstm", type=str)
    parser.add_argument("--device", default="cuda")

    # use for lstm
    parser.add_argument("--hidden_size", default=256)
    parser.add_argument("--num_layers", default=5)
    parser.add_argument("--output_size", default=5)
    parser.add_argument("--bidirectional", default=True)
    parser.add_argument("--dropout", default=0.25)

    args = parser.parse_args()

    train_data = train_data.map(
        tokenize_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length}
    )
    test_data = test_data.map(
        tokenize_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length}
    )

    train_valid_data = train_data.train_test_split(test_size=args.test_size)
    train_data = train_valid_data["train"]
    valid_data = train_valid_data["test"]

    special_tokens = ["<unk>", "<pad>"]

    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["tokens"],
        min_freq=args.min_freq,
        specials=special_tokens,
    )

    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)

    train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

    train_data = train_data.with_format(type="torch", columns=["ids", "label"])
    valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label"])

    train_data_loader = get_data_loader(train_data, args.batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, args.batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, args.batch_size, pad_index)

    vocab_size = len(vocab)
    output_dim = len(train_data.unique("label"))

    if args.model == "cnn":
        model = CNN(
            vocab_size,
            args.embedding_dim,
            args.n_filters,
            args.filter_sizes,
            output_dim,
            args.dropout_rate,
            pad_index,
        )
    elif args.model == "lstm":
        model = LSTM(
            vocab_size,
            args.hidden_size,
            args.num_layers,
            output_dim,
            args.bidirectional,
            args.dropout,
        )

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    metrics = collections.defaultdict(list)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    for epoch in range(args.n_epochs):
        train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, args.device)
        valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, args.device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)

        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    # Evaluate on test set
    test_loss, test_acc = evaluate(test_data_loader, model, criterion, args.device)
    print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")


    # Draw metrics
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_losses"], label="train")
    plt.plot(metrics["valid_losses"], label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_accs"], label="train")
    plt.plot(metrics["valid_accs"], label="valid")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.savefig(f"{args.model}-metrics.pdf")
