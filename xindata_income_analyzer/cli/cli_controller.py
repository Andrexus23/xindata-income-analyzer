"""CLI Controller."""
import cmd
from typing import Optional
from xindata_income_analyzer.model.text_generator import TextGenerator

class CLIController(cmd.Cmd):
    """Контроллер CLI."""
    
    def __init__(self, llm_interface=None, completekey='tab', stdin=None, stdout=None):
        super().__init__(completekey, stdin, stdout)
        self._llm_interface: Optional[TextGenerator] = llm_interface


    def default(self, line: str):
        """Обработка любых входящих запросов."""
        if not self._llm_interface:
            print("Модель не инициализирована.")
            return
        response: str = self._llm_interface.generate_text(line)
        print('Ответ: {0}'.format(response))
        
    def emptyline(self):
        """Обработка пустой строки."""
        return cmd.Cmd.emptyline(self)
    
    def precmd(self, line: str):
        return cmd.Cmd.precmd(self, line)
    
    def postcmd(self, stop, line: str):
        return cmd.Cmd.postcmd(self, stop, line)
    
    def do_EOF(self, line: str):
        """Т.к. do_EOF() вернёт True, нажатие Ctrl-D завершит работу интерпретатора."""
        return True

    