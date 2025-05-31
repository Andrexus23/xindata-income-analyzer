from transformers import pipeline

# Загрузить модель и токенизатор (например, GPT-2 или другую)
generator = pipeline('text-generation', model='gpt2')

prompt = "Freelancer earnings analysis shows that"
results = generator(prompt, max_length=100, num_return_sequences=1)

print(results[0]['generated_text'])