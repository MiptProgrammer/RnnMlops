import torch
import torch.nn as nn

from utils_rnn import load_data, letter_to_tensor, line_to_tensor, random_training_example, category_from_output, \
    N_LETTERS
from rnnmodel import RNN



def train():


    category_lines, all_categories = load_data()
    n_categories = len(all_categories)

    n_hidden = 128
    rnn = RNN(N_LETTERS, n_hidden, n_categories)

    # one step
    input_tensor = letter_to_tensor('A')
    hidden_tensor = rnn.init_hidden()

    output, next_hidden = rnn(input_tensor, hidden_tensor)
    # print(output.size())
    # print(next_hidden.size())

    # whole sequence/name
    input_tensor = line_to_tensor('Albert')
    hidden_tensor = rnn.init_hidden()

    output, next_hidden = rnn(input_tensor[0], hidden_tensor)

    # print(output.size())
    # print(next_hidden.size())

    #

    print(category_from_output(output, all_categories))

    criterion = nn.NLLLoss()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    def train_loop(line_tensor, category_tensor):
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, loss.item()

    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = 100000
    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

        output, loss = train_loop(line_tensor, category_tensor)
        current_loss += loss

        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i + 1) % print_steps == 0:
            guess = category_from_output(output, all_categories)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {line} / {guess} {correct}")

    print(rnn)
    torch.save(rnn.state_dict(), '../my_model.pth')


