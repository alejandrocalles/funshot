import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import StrEnum
import funshot.functions as functions

def load_model(model_id: str):
    print(f"Loading model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="auto"
    )
    print("Model loaded successfully")
    return tokenizer, model

def extract_python_code(text: str) -> str | None:
    """Extracts python code block from string."""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

class EvaluationResult(StrEnum):
    CORRECT_ANSWER = "correct answer"
    WRONG_ANSWER = "wrong answer"
    RUNTIME_ERROR = "runtime error"
    EXECUTION_ERROR = "execution error"
    SYNTAX_ERROR = "syntax error"
    CODE_NOT_FOUND_ERROR = "code not found error"

    def __init__(self, _):
        self.message = None

    def set_message(self, message: str):
        self.message = message
    
    def is_correct(self) -> bool:
        match self:
            case EvaluationResult.CORRECT_ANSWER:
                return True
            case _:
                return False

def evaluate(
    generation_config: dict,
    prompt_config: dict,
    tokenizer,
    model,
    function: functions.Function) -> EvaluationResult:
    """
    Evaluate a model's capabilities by asking it to code a python function and
    checking the function's correctness.
    
    :param generation_config: The configuration for generating the answer.
    :type generation_config: dict
    :param prompt_config: The configuration for creating the prompt.
    :type prompt_config: dict
    :param tokenizer: A tokenizer instantiated from `transformers.AutoTokenizer.from_pretrained`
    :param model: A model instantiated from `transformers.AutoModelforCausalLM.from_pretrained`
    :param function: The reference function to be used for evaluation.
    :type function: functions.Function
    """

    table = function.generate_formatted_table(**prompt_config['table_format'])
    prompt = prompt_config['user_prompt'].format(table, prompt_config['function_name'])

    messages = [
        {"role": "system", "content": prompt_config['system_prompt']},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=generation_config['max_new_tokens'],
            temperature=generation_config['temperature'],
            do_sample=True
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    code = extract_python_code(response)

    if code is None:
        print("❌ Failed to extract code block from model response.")
        print(f"Response: {response[:100]}")
        return EvaluationResult.CODE_NOT_FOUND_ERROR
    
    local_scope = {}
    try:
        exec(code, {}, local_scope)
        generated_function = local_scope[prompt_config['function_name']] 
        if function.test_generated_function(generated_function=generated_function):
            return EvaluationResult.CORRECT_ANSWER
        else:
            return EvaluationResult.WRONG_ANSWER
    except SyntaxError as e:
        print(f"❌ Syntax error in generated code: {e}")
        return EvaluationResult.SYNTAX_ERROR
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return EvaluationResult.EXECUTION_ERROR

