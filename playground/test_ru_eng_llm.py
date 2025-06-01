from xindata_income_analyzer.model.text_generator import TextGenerator

if __name__ == "__main__":
    generator = TextGenerator(model_name='/home/andrey/test-tasks/preptrained_models/updated_rugpt3small_based_on_gpt2')
    prompt = "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?"
    generated = generator.generate_text(prompt, max_length=150, temperature=0.8)
    print(f"Сгенерированный текст:\n{generated}")