from llama_model.model import LlamaModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("--model_path", type=str, default="./Llama-2-7b-chat-hf", help="Path to BASE_MODEL")
parser.add_argument("--data_len", type=int, default=100, help="number of data accuracy")
args = parser.parse_args()
model = LlamaModel(BASE_MODEL=args.model_path, batch_size=args.batch)
model.accuracy_check(data_len=100)