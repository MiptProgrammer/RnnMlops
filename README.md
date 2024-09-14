## RNN in PyTorch

Implement a Recurrent Neural Network (RNN) from scratch in PyTorch! This guide briefly explains the theory and different kinds of applications of RNNs. Then, we implement an RNN to perform name classification.

## Watch the Tutorial

[![Alt text](https://img.youtube.com/vi/WEV61GmmPrk/hqdefault.jpg)](https://youtu.be/WEV61GmmPrk)

## Resources

- **Download the data:** [data.zip](https://download.pytorch.org/tutorial/data.zip)

### Further Readings

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
- [Recurrent Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#architecture) by Shervine Amidi
- [PyTorch Char-RNN Classification Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

## Основные задачи проекта

- **Контейнеризация и развертывание:** Использование Docker и Docker Compose для создания и управления контейнерами, что позволяет легко развернуть приложение на любом сервере или облаке.
- **Обучение модели:** Разработка и обучение RNN для работы с текстовыми данными, поддержка многоязычности через использование различных наборов данных.
- **Инференс и использование модели:** Обеспечение удобного интерфейса для предсказаний на новых данных через инференс-скрипты.
- **Автоматизация:** Автоматизация процессов разработки, обучения и развертывания с помощью скриптов и конфигурационных файлов.

Проект предназначен для тех, кто интересуется применением MLOps в реальных задачах машинного обучения и контейнеризацией ML моделей.

## Структура проекта

```plaintext
.
├── Dockerfile
├── README.Docker.md
├── README.md
├── code
│   ├── init.py
│   ├── infer.py
│   ├── rnnmodel.py
│   ├── train.py
│   └── utils_rnn.py
├── commands.py
├── compose.yaml
├── data
│   └── names
│       ├── Arabic.txt
│       ├── Chinese.txt
│       ├── Czech.txt
│       ├── Dutch.txt
│       ├── English.txt
│       ├── French.txt
│       ├── German.txt
│       ├── Greek.txt
│       ├── Irish.txt
│       ├── Italian.txt
│       ├── Japanese.txt
│       ├── Korean.txt
│       ├── Polish.txt
│       ├── Portuguese.txt
│       ├── Russian.txt
│       ├── Scottish.txt
│       ├── Spanish.txt
│       └── Vietnamese.txt
└── requirements.txt

  
