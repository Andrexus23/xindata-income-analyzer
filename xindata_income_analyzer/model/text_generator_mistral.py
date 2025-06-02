from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TextGeneratorMistral:
    def __init__(self, model_name="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", device=None):
        """
        Инициализация генератора текста.
        :param model_name: имя модели на Hugging Face Hub
        :param device: устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Загрузка модели на устройство: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # автоматически распределяет модель по доступным GPU/CPU
            torch_dtype=torch.float16,  # для ускорения и экономии памяти
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)

    def generate_text(self, prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.9) -> str:
        """
        Генерация текста по запросу.
        :param prompt: входной текст
        :param max_new_tokens: максимальное количество генерируемых токенов
        :param temperature: параметр температуры для разнообразия
        :param top_p: параметр nucleus sampling
        :return: сгенерированный текст
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text