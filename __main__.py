from xindata_income_analyzer.config.config import get_config
from xindata_income_analyzer.cli.cli_controller import CLIController
from xindata_income_analyzer.model.text_generator import TextGenerator

if __name__ == "__main__":
    llm_interface: TextGenerator = TextGenerator(model_name=get_config().model_settings.new_model_path)
    cli = CLIController(llm_interface=llm_interface)
    cli.cmdloop()

