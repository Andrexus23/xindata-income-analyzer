from xindata_income_analyzer.config.config import get_config
from xindata_income_analyzer.cli.cli_controller import CLIController
from xindata_income_analyzer.model.text_generator_gpt2 import TextGeneratorGPT2

if __name__ == "__main__":
    llm_interface: TextGeneratorGPT2 = TextGeneratorGPT2(model_name=get_config().model_settings.new_model_path)
    cli = CLIController(llm_interface=llm_interface)
    cli.cmdloop()

