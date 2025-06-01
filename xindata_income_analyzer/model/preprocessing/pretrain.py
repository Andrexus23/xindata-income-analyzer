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
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=5000,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="no",  # валидация на этапе обучения не предполагается
        fp16=True,  # GPU
    )
    
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_model_file)
    trainer.save_pretrained(output_model_file)