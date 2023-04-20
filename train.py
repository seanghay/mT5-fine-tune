from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from khmernltk import sentence_tokenize
import evaluate
import torch
import numpy as np

MODEL_NAME = "google/mt5-base"
DATASET_NAME = "seanghay/koh-233k"
OUTPUT_DIR = "outputs/mt5-base-koh-233k"

NUM_PROC=30
SEED = 10
MAX_INPUT_LENGTH = 2048
MAX_TARGET_LENGTH = 256
INPUT_PREFIX="summarize:"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=True).to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = evaluate.load("rouge")

model.config.use_cache = False

print(f"device: {device}")

ds = load_dataset(DATASET_NAME, split="train", use_auth_token=True)
ds = ds.rename_column("title", "summary").rename_column("content", "document")
ds = ds.shuffle(seed=SEED)
raw_datasets = ds.train_test_split(test_size=0.1)
print("raw_datasets:")
print(raw_datasets)

def check_tokenizer():
  input_str = ds[0]["summary"]
  labels = tokenizer(input_str).input_ids
  return tokenizer.decode(labels, skip_special_tokens=True) == input_str

print(f"tokenizer can decode and encode: {check_tokenizer()}")

def preprocess_function(examples):
    inputs = [INPUT_PREFIX + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=NUM_PROC).remove_columns(["summary", "document"])

print("tokenized_datasets:")
print(tokenized_datasets)

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  decoded_preds = ["\n".join(sentence_tokenize(pred.strip())) for pred in decoded_preds]
  decoded_labels = ["\n".join(sentence_tokenize(label.strip())) for label in decoded_labels]

  result = metric.compute(
    predictions=decoded_preds, 
    references=decoded_labels, 
    use_stemmer=True, 
    tokenizer=tokenizer.tokenize
  )

  result = {key: value * 100 for key, value in result.items()}
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)
    
  return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    fp16=True,
    learning_rate=5e-5,
    num_train_epochs=5,
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
trainer.push_to_hub()