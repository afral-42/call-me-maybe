cd ..

mkdir -p data/output

uv run python -m src --functions_definition tests/defs/functions_integer.json --input tests/prompts/prompts_integer.json --output data/output/results_integer.json

uv run python -m src --functions_definition tests/defs/functions_number.json --input tests/prompts/prompts_number.json --output data/output/results_number.json

uv run python -m src --functions_definition tests/defs/functions_string.json --input tests/prompts/prompts_string.json --output data/output/results_string.json

uv run python -m src --functions_definition tests/defs/functions_boolean.json --input tests/prompts/prompts_boolean.json --output data/output/results_boolean.json