import dataclasses
import typing
import random

@dataclasses.dataclass
class Function:
    name: str
    description: str
    apply: typing.Callable[[int, int], int]

    def __call__(self, x: int, y: int) -> int:
        return self.apply(x, y)
    
    def generate_formatted_table(self, num_rows: int, row_format: str, header: str | None = None) -> str:
        lines: list[str] = [] if header is None else [header]
        table = ValueTable(self, num_rows=num_rows)
        lines += [row_format.format(*row.values()) for row in table.data]
        
        return "\n".join(lines)

    def test_generated_function(self, generated_function, count=10, low=100, high=999) -> bool:
        for i in range(count):
            x = random.randint(low, high)
            y = random.randint(low, high)
            
            expected = self(x, y)
            actual = generated_function(x, y)
            
            if actual != expected:
                print(f"Failed test {i + 1}/{count}: expected {expected} for f({x}, {y}), got {actual}.")
                return False
        return True

class ValueTable:
    def __init__(self, function: Function, num_rows: int, low: int = 1, high: int = 99):
        """Instantiate a value table with randomized values."""
        data: list[dict[str, int]] = []
        for _ in range(num_rows):
            x = random.randint(low, high)
            y = random.randint(low, high)

            output = function(x, y)
            data += [{
                "x": x,
                "y": y,
                "output": output
            }]
        self.data = data
        self.function = function

all_functions: list[Function] = [
    Function(
        name="Multiplication",
        description="x * y",
        apply=lambda x, y: x * y,
    ),
    Function(
        name="Addition",
        description="x + y",
        apply=lambda x, y: x + y
    ),
    Function(
        name="Strange Modulo",
        description="(x % y + 1)",
        apply=lambda x, y: (x % y) + 1 if y != 0 else 0
    )
]
