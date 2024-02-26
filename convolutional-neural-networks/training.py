import argparse
import importlib
from d2l import torch as d2l
from config import init_cnn


def train_model(args):
    model_module=importlib.import_module(args.model_path)
    model=getattr(model_module, args.model_name)
    data = d2l.FashionMNIST(batch_size=args.batch_size, resize=(32, 32))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn())
    trainer = d2l.Trainer(max_epochs=args.max_epoch, num_gpus=args.num_gpus)
    trainer.fit(model, data)


def get_parses():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help='Specify the model path')
    parser.add_argument("--model_name", type=str, help='Specify the model name')
    parser.add_argument("--arch", type=list, help='Architecture of the custom model (list of tuples: number of convolutions, output channels)')
    parser.add_argument("--batch_size", type=int, default=64, help='Specify the batch size')
    parser.add_argument("--max_epoch", type=int, default=10 ,help='Specify the maximum number of epochs')
    parser.add_argument("--num_gpus", type=int, default=1, help='Specify the number of GPUS')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=get_parses()
    train_model(args)
