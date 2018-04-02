import torch.nn as nn
from torch.autograd import Variable


class Speech_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size)
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i20 = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftMax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i20(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def make_video_training_data(videos):
    pass


def make_speech_training_data(speech):
    pass

if __name__ == "__main__":
    videos = []
    speech = []
    video_data = make_video_training_data(videos)
    speech_data = make_speech_training_data(speech)
