from xindata_income_analyzer.cli_controller import CLIController
from xindata_income_analyzer.model.text_generator import TextGenerator

path = '/home/andrey/test-tasks/preptrained_models/updated_rugpt3small_based_on_gpt2'
if __name__ == "__main__":
    llm_interface: TextGenerator = TextGenerator(model_name=path)
    cli = CLIController(llm_interface=llm_interface)
    cli.cmdloop()

