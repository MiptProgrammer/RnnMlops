import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils_rnn import ALL_LETTERS, N_LETTERS, category_from_output
from utils_rnn import load_data, letter_to_tensor, line_to_tensor, random_training_example
from rnnmodel import RNN

def infer(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():

        category_lines, all_categories = load_data()
        n_categories = len(all_categories)
        n_hidden = 128

        line_tensor = line_to_tensor(input_line)
        rnn = RNN(N_LETTERS, n_hidden, n_categories)
        rnn.load_state_dict(torch.load('../my_model.pth'))

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output, all_categories)
        print(guess)

