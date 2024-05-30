import torch
import transformers
import os 
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, GenerationConfig
from peft import PeftModel
from sklearn.model_selection import train_test_split
from peft import (
    LoraConfig,
    get_peft_model
)
import time
import csv
from data import optimize_text


class LlamaModel:
    def __init__(
        self,
        BASE_MODEL = "./Llama-2-7b-chat-hf",
        DEVICE = "cpu",
        batch_size = 16,
        test_data_path = "./data/test.txt",
        train_data_path = "./data/emotions.csv",
        val_data_path = "./data/val.txt"
    ):
        self.DEVICE = DEVICE
        self.BASE_MODEL = BASE_MODEL
        self.R = 256
        #max number of every emotions in val and test data (the excess will be sent to train_data)
        self.MAX_NUM_EMOTION_DATA = 5000
        try:
            self.ADAPTER_MODELS_ALL = os.listdir(f"emotion_detecting/batch_{batch_size}_r_{self.R}_optimized_data_{self.MAX_NUM_EMOTION_DATA}/")
            ADAPTER_MODELS = []
            for model in self.ADAPTER_MODELS_ALL:
                if "checkpoint" in model:
                    ADAPTER_MODELS.append(f"emotion_detecting/batch_{batch_size}_r_{self.R}_optimized_data_{self.MAX_NUM_EMOTION_DATA}/"+model)
            self.ADAPTER_MODEL = ADAPTER_MODELS[-1]
        except:
            print("Check your output_dir")
        self.tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, legacy="False")
        self.INSTRUCTION = "Which of the emotions: sadness, joy, fear, anger, love, surprise, is more suitable for this text?"
        self.CUTOFF_LEN = 5000
        self.test_data_path = test_data_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_data, self.test_data, self.val_data = self.optimize_data()
        torch.cuda.empty_cache()

    def optimize_data(self):
        test_data = self.get_data(self.test_data_path)
        train_data = self.get_csv_data(self.train_data_path)
        val_data = self.get_data(self.val_data_path)
        check_val_data = self.check_data(val_data)
        check_test_data = self.check_data(test_data)
        #test data
        for key,value in check_test_data.items():
            if value > self.MAX_NUM_EMOTION_DATA:
                check = value
                i = 0
                while i < len(test_data):
                    if check == self.MAX_NUM_EMOTION_DATA:
                        break
                    if test_data[i]["emotion"] == key:
                        train_data.append(test_data[i])
                        test_data.remove(test_data[i])
                        check -= 1
                    else:
                        i += 1
        #val data
        for key,value in check_val_data.items():
            if value > self.MAX_NUM_EMOTION_DATA:
                check = value
                i = 0
                while i < len(val_data):
                    if check == self.MAX_NUM_EMOTION_DATA:
                        break
                    if val_data[i]["emotion"] == key:
                        train_data.append(val_data[i])
                        val_data.remove(val_data[i])
                        check -= 1
                    else:
                        i += 1
        return train_data, test_data, val_data

    def check_data(self, data = []):
        if data == []:
            data = self.test_data
        result = {}
        for dict in data:
            try:
                result[dict["emotion"]] += 1
            except:
                result[dict["emotion"]] = 0
        return result
    
    def get_data(self, data_path):
        with open(data_path, 'r') as data_file:
            reader = data_file.readlines()
            data = []
            i = 0
            for line in reader:
                data.append({})
                data[i]["emotion"] = line.split(";")[1].split("\n")[0]
                data[i]["content"] = line.split(";")[0]
                i += 1
        return data
    
    def get_csv_data(self, csv_data_path):
        with open(csv_data_path, mode='r') as infile:
            reader = csv.reader(infile)
            mylist = list(reader)
        data = []
        for i in range(len(mylist)):
            data.append({})
            data[i]["emotion"] = mylist[i][0]
            data[i]["content"] = mylist[i][1]
        return data

    def get_model_for_infer(self):
        model = LlamaForCausalLM.from_pretrained(
            self.BASE_MODEL,
            device_map=self.DEVICE
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
        text = optimize_text(text)
        prompt = f"{self.INSTRUCTION}\nText:\n{text}\nemotion: "
        response = self.generate_response(prompt, model)
        decoded_output = self.tokenizer.decode(response.sequences[0], skip_special_tokens=True)
        decoded_output_lines = decoded_output.split("\n")
        #print(decoded_output_lines)
        for line in decoded_output_lines:
            if "emotion:" in line:
                emotion = line
                break
        return emotion
    
    def generate_prompt(self, sample):
        tokenizer = self.tokenizer_for_train()
        prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nemotion: "
        full_prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nemotion: {sample['emotion']}"

        # If the length of full_prompt is greater than CUTOFF_LEN, remove the last few sentences of the text,
        # until the length is less than CUTOFF_LEN
        if len(tokenizer(full_prompt)["input_ids"]) > self.CUTOFF_LEN:
            sentences = sample['content'].split(". ")   
            while True:
                sentences = sentences[:-1]
                text = ". ".join(sentences)
                prompt = f"{self.INSTRUCTION}\n:\n{sample['content']}\nemotion: "
                full_prompt = f"{self.INSTRUCTION}\nText:\n{sample['content']}\nemotion: {sample['emotion']}"
                if len(tokenizer(full_prompt)["input_ids"]) < self.CUTOFF_LEN:
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

    def measures_check(self, data_len = 5000):
        if data_len == 5000:
            data_len = self.MAX_NUM_EMOTION_DATA
        model = self.get_model_for_infer()
        data = self.test_data
        if data_len > len(data):
            data_len = len(data)
            
        print("Data len ", data_len)
        acc_good_strings = 0
        wrong_answers = 0
        emoji_dict = {
            "sadness":"😢😔💔",
            "joy":"😃😊😂😋",
            "fear":"😐😬😨",
            "anger":"😠😤😡",
            "love":"😍",
            "surprise":"😂😃"
        }
        emotion_arr = ["joy", "love", "surprise", "sadness", "fear", "anger"]
        st = time.time()
        check_process = data_len*0.1
        emotion_metrics = {}
        for emotion in emotion_arr:
            emotion_metrics[emotion] = {}
            emotion_metrics[emotion]["TP"] = 0
            emotion_metrics[emotion]["TN"] = 0
            emotion_metrics[emotion]["FP"] = 0
            emotion_metrics[emotion]["FN"] = 0
        for i in range(data_len):
            if i == check_process:
                print(f"Now number:{i} {(i//(data_len*0.1))*10}%")
                check_process += data_len*0.1
            emotion = self.generate_emotion(data[i]["content"], model).lower()
            if emotion != "emotion: ":
                acc_good_strings += 1
                emotion = emotion.split()[-1]
                for key, value in emoji_dict.items():
                    if emotion in value:
                        emotion = key 
                if emotion in emotion_arr:
                    if emotion == data[i]["emotion"]:
                        emotion_metrics[emotion]["TP"] += 1
                        for emo in emotion_arr:
                            if emo != emotion:
                                emotion_metrics[emo]["TN"] += 1
                        #print(i, "True", data[i]["emotion"], emotion)
                    else:
                        emotion_metrics[emotion]["FP"] += 1
                        emotion_metrics[data[i]["emotion"]]["FN"] += 1
                else:
                    wrong_answers += 1
                    #print(i, "Wrong", data[i]["emotion"], emotion)
        elapsed_time = time.time() - st
        for_print = {
            "Not empty answers: ": f"{acc_good_strings}/{data_len}",
            "Wrong answers: ": f"{wrong_answers}/{data_len}",
            "Execution time:": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        }
        accuracy_top = 0
        accuracy_down = 0
        for emotion in emotion_arr:
            TP = emotion_metrics[emotion]["TP"]
            TN = emotion_metrics[emotion]["TN"]
            FP = emotion_metrics[emotion]["FP"]
            FN = emotion_metrics[emotion]["FN"]
            print(emotion, f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")
            accuracy_top += TP+TN
            accuracy_down += TP+TN+FP+FN
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            F1 = 2*(precision * recall) / (precision + recall)
            for_print[f"{emotion}_Precision"] = precision
            for_print[f"{emotion}_Recall"] = recall
            for_print[f"{emotion}_F1"] = F1

        for_print["Accuracy"] = accuracy_top / accuracy_down

        for key, value in for_print.items():
            print(key, value)

    def tokenizer_for_train(self):
        tokenizer = self.tokenizer
        # set a token for padding, that is, adding to those sequences from the batch that are shorter,
        # than the maximum length of a sequence so that all sequences end up being the same length
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        tokenizer.load_in_8bit_fp32_cpu_offload=True
        return tokenizer

    def train(self, batch_size = 8, output_dir = "emotion_detecting"):
        tokenizer = self.tokenizer_for_train()
        print(len(self.train_data))
        model = LlamaForCausalLM.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=torch.float16,
            device_map=self.DEVICE,
        )
        # Data preprocessing (receiving a prompt for each example from the dataset and subsequent tokenization)
        train_data = list(map(self.generate_and_tokenize_prompt, self.train_data))
        test_data = list(map(self.generate_and_tokenize_prompt, self.val_data))

        # data_collator is needed to form a batch (padding, assembling batch elements into one tensor,
        # convert numpy arrays or lists to tensors torch.LongTensor)
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, label_pad_token_id=-100
        )

        #adapters

        # Dimension of adapter matrices
        # For example, if the original weight matrix is ​​4096 x 4096, then the matrices we add are
        # have dimensions 4096 x LORA_R and LORA_R x 4096.
        LORA_R = self.R

        # After multiplying by the matrix of adapter weights, divide the vector components by LORA_R and multiply by LORA_ALPHA
        LORA_ALPHA = 32
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
        OUTPUT_DIR = f"{output_dir}/batch_{batch_size}_r_{LORA_R}_optimized_data_{self.MAX_NUM_EMOTION_DATA}"
        BATCH_SIZE = batch_size
        TRAIN_EPOCHS = 3
        MICRO_BATCH_SIZE = 1
        GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
        LEARNING_RATE = 3e-4


        steps = 500
        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=steps,
            #max_steps=1000,
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=steps,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=steps,
            save_steps=steps,
            output_dir=OUTPUT_DIR,
            save_total_limit=1,
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
        