from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/Meta-Llama-3-8B-nli-random-standardized-random-names")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/Meta-Llama-3-8B-nli-random-standardized-random-names")

training_args = TrainingArguments(output_dir="./fine-tuned", num_train_epochs=5)
trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

trainer.train()
model.save_pretrained("./fine-tuned")