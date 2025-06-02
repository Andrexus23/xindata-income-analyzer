import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGeneratorGPT2:
    def __init__(self, model_name='ai-forever/rugpt3small_based_on_gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "do_sample": True,
            }

            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return prompt