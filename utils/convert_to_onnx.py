import torch

from ..model.model import Net


def main():
  pytorch_model = Net()
  pytorch_model.load_state_dict(torch.load('../model/MNIST_model.pt'))
  pytorch_model.eval()
  dummy_input = torch.zeros(280 * 280 * 4)
  torch.onnx.export(pytorch_model, dummy_input, '../model/onnx_model.onnx', verbose=True)


if __name__ == '__main__':
  main()