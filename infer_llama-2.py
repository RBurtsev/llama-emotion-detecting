from llama_model.model import LlamaModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("--model_path", type=str, default="./Llama-2-7b-chat-hf", help="Path to BASE_MODEL")
args = parser.parse_args()
print(args)
model = LlamaModel(BASE_MODEL=args.model_path, batch_size=args.batch)
text = "i am ever feeling nostalgic about the fireplace i will know that it is still on the property"
print(model.generate_emotion(text = text, model=model.get_model_for_infer()))