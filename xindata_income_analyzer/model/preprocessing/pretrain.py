"""Предобучение моделей."""
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling

from xindata_income_analyzer.model.preprocessing import constants


def pretrain_model(model_name: str, dataset_file: str, output_model_file: str):
    """Предобучить модель.

    Args:
        model_name (str): путь к исходной модели.
        dataset_file (str): путь к датасету для обучения.
        output_model_file (str): выходной файл, куда будет записана обученная модель.
    """
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_file,
        block_size=constants.BLOCK_SIZE,  # https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2
    )
    data_collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # для GPT каузальное языковое моделирование
    )
    
    training_args: TrainingArguments = TrainingArguments(
        output_dir=output_model_file,
        overwrite_output_dir=True,
        num_train_epochs=constants.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=constants.PER_DEVICE_TRAIN_BATCH_SIZE,
        save_steps=constants.SAVE_STEPS,
        save_total_limit=constants.SAVE_TOTAL_LIMIT,
        logging_steps=constants.LOGGING_STEPS,
        learning_rate=constants.LEARNING_RATE,
        weight_decay=constants.WEIGHT_DECAY,
        warmup_steps=constants.WARMUP_STEPS,
        eval_strategy=constants.EVALUATION_STRATEGY,  # валидация на этапе обучения не предполагается
        fp16=constants.FP16,  # GPU
    )
    
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_model_file)
    tokenizer.save_pretrained(output_model_file)