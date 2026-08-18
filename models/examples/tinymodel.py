
import torch

class TinyModel(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":

    tinymodel = TinyModel()

    print('The model:')
    print(tinymodel)

    print('\n\nJust one layer:')
    print(tinymodel.linear2)

    print('\n\nModel params:')
    for param in tinymodel.parameters():
        print(param)

    print('\n\nLayer params:')
    for param in tinymodel.linear2.parameters():
        print(param)
