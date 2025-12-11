import tqdm
import torch
from funshot.helpers import load_config
from funshot.generation import evaluate, load_model
from funshot.functions import all_functions
import random
import datetime
import pathlib
import yaml

def run(config_name: str):
    config = load_config(config_name)
    tokenizer, model = load_model(model_id=config['model']['id'])

    output_path = pathlib.Path.cwd().joinpath(
        config['output_dir'],
        datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
    ).resolve()

    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path.joinpath('config_dump.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    results = []

    for function in all_functions:
        torch.manual_seed(config['generation']['pytorch_seed'])
        print(f"TARGET: {function.name} ({function.description})")

        num_correct = 0
        num_attempts = config['generation']['num_attempts']

        for i in range(num_attempts):
            random.seed(config['random_seed'] + i)

            result = evaluate(
                generation_config=config['generation'],
                prompt_config=config['prompt'],
                tokenizer=tokenizer,
                model=model,
                function=function
            )

            num_correct += result.is_correct()
        
        accuracy = num_correct / num_attempts
        print(f"Accuracy: {accuracy * 100:.0f}% ({num_correct} / {num_attempts})")
        results += [{
            "model": config['model']['id'],
            "function": function.description,
            "num_correct": num_correct,
            "num_attempts": num_attempts
        }]
    
    with open(output_path.joinpath('results.yaml'), 'w') as f:
        yaml.safe_dump(results, f)




