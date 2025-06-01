from xindata_income_analyzer.model.preprocessing.pretrain import pretrain_model


if __name__ == "__main__":
    pretrain_model(
        "ai-forever/rugpt3small_based_on_gpt2", 
        '/home/andrey/test-tasks/xindata-income-analyzer/playground/this.txt', 
        '/home/andrey/test-tasks/preptrained_models/updated_rugpt3small_based_on_gpt2',
    )
