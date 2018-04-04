import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class Speech_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Speech_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i20 = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i20(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class Video_CNN(nn.Module):
    def __init__(self):
        super(Video_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=18), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Dropout(.2), nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=8), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(.2), nn.MaxPool1d(2))
        self.fc = nn.Linear(32 * 60, 32 * 10)
        self.fc2 = nn.Linear(32 * 10, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 32 * 60)
        out = self.fc(out)
        out = self.fc2(out)

        return out


def make_video_training_data(videos) -> np.ndarray:
    pass


def make_speech_training_data(speech) -> np.ndarray:
    pass


def train_NN(NN: nn.Module, dataset: TensorDataset, loss_func,
             out_file_name: str):
    num_epochs = 100
    loader = DataLoader(
        dataset, 100, shuffle=True, num_workers=8, pin_memory=True)
    best_epoch = None
    best_train_loss = None

    for epoch in num_epochs:
        for sample in DataLoader:
            train_x = sample[0]
            train_y = sample[1]
            train_y_hat = NN(train_x)
            train_loss = loss_func(train_y_hat, train_y)

            if not best_epoch or train_loss < best_train_loss:
                best_epoch = epoch
                best_train_loss = train_loss
    with open(out_file_name, 'w') as out:
        out.write('Best epoch: {0}\n Best train loss: {1}'.format(
            best_epoch, best_train_loss))


if __name__ == "__main__":
    videos = []
    speech = []
    video_data = make_video_training_data(videos)
    speech_data = make_speech_training_data(speech)
    labels = []
    n_hidden = 128
    speech_rnn = Speech_RNN(13, n_hidden, 2).cuda()
    video_cnn = Video_CNN().cuda()
    video_dataset = TensorDataset(video_data, labels).cuda()
    speech_dataset = TensorDataset(speech_data, labels)
    train_NN(video_cnn, video_dataset, nn.MSELoss(), 'video_cnn_stats.txt')
    train_NN(speech_rnn, speech_dataset, nn.NLLLoss(), 'speech_cnn_stats.txt')
