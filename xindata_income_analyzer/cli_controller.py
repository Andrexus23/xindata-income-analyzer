"""CLI Controller."""
import cmd
import shlex
from typing import Optional
from xindata_income_analyzer.model.text_generator import TextGenerator

class CLIController(cmd.Cmd):
    
    def __init__(self, llm_interface=None, completekey='tab', stdin=None, stdout=None):
        super().__init__(completekey, stdin, stdout)
        self._llm_interface: Optional[TextGenerator] = llm_interface
        
    def prompt(self, prompt: str):
        """Обработка prompt."""
        if not self._llm_interface:
            print("Модель не инициализирована.")
            return
        response: str = self._llm_interface.generate_text(prompt)
        print('Ответ: {0}'.format(response))
    