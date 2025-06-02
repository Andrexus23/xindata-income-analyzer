from xindata_income_analyzer.config.config import get_config
from xindata_income_analyzer.cli.cli_controller import CLIController
from xindata_income_analyzer.model.text_generator_gpt2 import TextGeneratorGPT2
from xindata_income_analyzer.model.text_generator_mistral import TextGeneratorMistral

if __name__ == "__main__":
    llm_interface: TextGeneratorMistral = TextGeneratorMistral(model_name='Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24')
    cli = CLIController(llm_interface=llm_interface)
    cli.cmdloop()

