import torch
import transformers
import csv
import os 
import emoji
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
from sklearn.model_selection import train_test_split
from peft import (
    LoraConfig,
    get_peft_model
)


class LlamaModel:
    def __init__(
        self,
        BASE_MODEL = "./Llama-2-7b-chat-hf",
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size = 16
    ):
        self.DEVICE = DEVICE
        self.BASE_MODEL = BASE_MODEL
        try:
            self.ADAPTER_MODELS_ALL = os.listdir(f"emotion_detecting/batch_{batch_size}/")
            ADAPTER_MODELS = []
            for model in self.ADAPTER_MODELS_ALL:
                if "checkpoint" in model:
                    ADAPTER_MODELS.append(f"emotion_detecting/batch_{batch_size}/"+model)
            self.ADAPTER_MODEL = ADAPTER_MODELS[-1]
        except:
            print("Check your output_dir")
        self.tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, legacy="False")
        self.INSTRUCTION = "Detect emotion from text."
        self.CUTOFF_LEN = 500

    def get_model_for_infer(self):
        model = LlamaForCausalLM.from_pretrained(
            self.BASE_MODEL,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(model, self.ADAPTER_MODEL, torch_dtype=torch.float16)
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model = model.eval()
        model = torch.compile(model)
        return model

    def generate_response(self, prompt, model):
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.DEVICE)
    
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            repetition_penalty=1.1,
        )
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
            )

    def generate_emotion(self, text, model):
        prompt = f"{self.INSTRUCTION}\nText:\n{text}\nEmotion: "
        response = self.generate_response(prompt, model)
        decoded_output = self.tokenizer.decode(response.sequences[0], skip_special_tokens=True)
        decoded_output_lines = decoded_output.split("\n")
        #print(decoded_output_lines)
        for line in decoded_output_lines:
            if "Emotion:" in line:
                emotion = line
                break
        return emotion
    
    def generate_prompt(self, sample):
        tokenizer = self.tokenizer_for_train()
        prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nEmotion: "
        full_prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nEmotion: {sample['sentiment']}"

        # If the length of full_prompt is greater than CUTOFF_LEN, remove the last few sentences of the text,
        # until the length is less than CUTOFF_LEN
        if len(tokenizer(full_prompt)["input_ids"]) > self.CUTOFF_LEN:
            sentences = sample['content'].split(". ")   
            while True:
                sentences = sentences[:-1]
                text = ". ".join(sentences)
                prompt = f"{self.INSTRUCTION}\n:\n{sample['content']}\nEmotion: "
                full_prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nEmotion: {sample['sentiment']}"
                if len(tokenizer(full_prompt)["input_ids"]) < CUTOFF_LEN:
                    break
        return prompt, full_prompt

    def tokenize(self, prompt, full_prompt, add_eos_token=True):
        tokenizer = self.tokenizer_for_train()
        result = tokenizer(
            full_prompt,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < self.CUTOFF_LEN
                and add_eos_token
        ):
            # if there is no special token at the end of the post, we add it
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        prompt_len = len(tokenizer(prompt)["input_ids"])
        labels = result["input_ids"].copy()
        labels = [-100 for _ in range(prompt_len)] + labels[prompt_len:]
        result["labels"] = labels

        return result

    def generate_and_tokenize_prompt(self, sample):
        prompt, full_prompt = self.generate_prompt(sample)
        tokenized_full_prompt = self.tokenize(prompt, full_prompt)
        return tokenized_full_prompt

    def accuracy_check(self, data_path = "./data/test.txt", data_len = 500):
        model = self.get_model_for_infer()
        with open(data_path, 'r') as data_file:
            reader = data_file.readlines()
            data = []
            i = 0
            for line in reader:
                data.append({})
                data[i]["sentiment"] = line.split(";")[1].split("\n")[0]
                data[i]["content"] = line.split(";")[0]
                i += 1
        
        if data_len > len(data):
            data_len = len(data)
            
        print("Data len ", data_len)
        accuracy = 0
        arr_accuracy_emotions = {}
        acc_good_string = []
        check_number = 0 
        emoji_dict = {
            "sadness":"😢😔💔",
            "joy":"😃😊😂😋",
            "fear":"😐😬😨",
            "anger":"😠😤😡",
            "love":"😍",
            "surprise":"😂😃"
        }
        for i in range(data_len):
            emotion = self.generate_emotion(data[i]["content"], model).lower()
            if emotion != "emotion: ":
                acc_good_string.append(data[i])
                emotion = emotion.split()[-1]
                check_number += 1
                if emotion == data[i]["sentiment"] or emotion in emoji_dict[data[i]["sentiment"]]:
                    accuracy += 1
                    try:
                        arr_accuracy_emotions[data[i]["sentiment"]] += 1
                    except:
                        arr_accuracy_emotions[data[i]["sentiment"]] = 1
                    #print(i, "True", data[i]["sentiment"], emotion)
                #else:
                    #print(i, "False", data[i]["sentiment"], emotion)    
        print(accuracy/check_number)
        print("Not empty answers: ", f"{len(acc_good_string)}/{data_len}")

    def tokenizer_for_train(self):
        tokenizer = self.tokenizer
        # set a token for padding, that is, adding to those sequences from the batch that are shorter,
        # than the maximum length of a sequence so that all sequences end up being the same length
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        tokenizer.load_in_8bit_fp32_cpu_offload=True
        return tokenizer

    def train(self, data_path = "./data/train.txt", batch_size = 8, output_dir = "emotion_detecting"):
        tokenizer = self.tokenizer_for_train()
        model = LlamaForCausalLM.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        with open(data_path, 'r') as data_file:
            reader = data_file.readlines()
            dataset = []
            i = 0
            for line in reader:
                dataset.append({})
                dataset[i]["sentiment"] = line.split(";")[1].split("\n")[0]
                dataset[i]["content"] = line.split(";")[0]
                i += 1
        train_data, test_data = train_test_split(dataset, test_size=0.1)
        # Data preprocessing (receiving a prompt for each example from the dataset and subsequent tokenization)
        train_data = list(map(self.generate_and_tokenize_prompt, train_data))
        test_data = list(map(self.generate_and_tokenize_prompt, test_data))

        # data_collator is needed to form a batch (padding, assembling batch elements into one tensor,
        # convert numpy arrays or lists to tensors torch.LongTensor)
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, label_pad_token_id=-100
        )

        #adapters

        # Dimension of adapter matrices
        # For example, if the original weight matrix is ​​4096 x 4096, then the matrices we add are
        # have dimensions 4096 x LORA_R and LORA_R x 4096.
        LORA_R = 8

        # After multiplying by the matrix of adapter weights, divide the vector components by LORA_R and multiply by LORA_ALPHA
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05

        # To which layers of the transformer will we add adapters, in this case - to the matrices in the self-attention layers
        # to calculate query and key.
        LORA_TARGET_MODULES = [
            "q_proj",
            "v_proj",
        ]

        # Create a configuration object based on adapter parameters
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        # Display information about the model's trained weights.
        model.print_trainable_parameters()
        OUTPUT_DIR = f"{output_dir}/batch_{batch_size}"
        BATCH_SIZE = batch_size
        TRAIN_EPOCHS = 3
        MICRO_BATCH_SIZE = 1
        GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
        LEARNING_RATE = 3e-4


        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=200,
            #max_steps=8000,
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=200,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=OUTPUT_DIR,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=test_data,
            args=training_arguments,
            data_collator=data_collator
        )

        model.config.use_cache = False

        # model compilation (to optimize training)
        model = torch.compile(model)
        torch.cuda.empty_cache()

        trainer.train()

        model.save_pretrained(OUTPUT_DIR)
        