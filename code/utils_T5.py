from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from random import sample



MODEL_NAME = "google/flan-t5-small" 
MAX_LENGTH = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
FEW_SHOT_EXAMPLES = [
    {
        "question": "Which of these is a fruit?",
        "choices": ["Carrot", "Banana", "Broccoli", "Potato"],
        "answer": "B"
    },
    {
        "question": "What animal is known for its stripes?",
        "choices": ["Elephant", "Zebra", "Lion", "Giraffe"],
        "answer": "B"
    }
]


def build_few_shot_prompt(row, examples=FEW_SHOT_EXAMPLES, num_shots=2):
    prompt = ""

    if num_shots > 0:
        selected_examples = sample(examples, min(num_shots, len(examples)))

        for ex in selected_examples:
            prompt += f"""Question: {ex['question']}
Options:
A) {ex['choices'][0]}
B) {ex['choices'][1]}
C) {ex['choices'][2]}
D) {ex['choices'][3]}
Answer: {ex['answer']}\n\n"""

    prompt += f"""Question: {row['clean_question']}
Options:
A) {row['choice_a']}
B) {row['choice_b']}
C) {row['choice_c']}
D) {row['choice_d']}
Answer:"""

    return {
        "input_text": prompt,
        "target_text": row["answer"].upper()
    }


def preprocess(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )
    
    targets = tokenizer(
        text_target=examples["target_text"],
        max_length=10,
        truncation=True,
        padding="max_length",
        return_tensors="np",
        add_special_tokens=False  
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    correct = 0
    invalid_predictions = 0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Nettoyage plus robuste
        pred = pred.strip().upper()[:1]
        label = label.strip().upper()[:1] if label else ""
        
        if pred in ['A', 'B', 'C', 'D'] and label in ['A', 'B', 'C', 'D']:
            if pred == label:
                correct += 1
        else:
            invalid_predictions += 1
    
    total = len(decoded_labels)
    accuracy = correct / total if total > 0 else 0
    invalid_rate = invalid_predictions / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "correct_count": correct,
        "total_count": total
    }




