import sys
import time
from src.inference import BatchInferenceEngine, InferenceException
from src.parsing import ParsingException
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained Inference Engine for Function Calling"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        begin = time.perf_counter()

        engine = BatchInferenceEngine.build_engine(
            input_path=args.input,
            functions_definition_path=args.functions_definition,
            output_path=args.output
        )

        prompt_count = len(engine.built_prompts)
        print(
            "Engine built successfully. Starting batch generation "
            f"for {prompt_count} prompts..."
        )

        engine.run()
        print("Generation completed!")

        print(f"Saving results to '{args.output}'...")
        engine.save_results()

        end = time.perf_counter()

        print(f"All done! Total execution time: {end - begin:.2f} seconds.")

    except ParsingException as e:
        print(f"\n[Parsing Error] {e}", file=sys.stderr)
        sys.exit(1)

    except InferenceException as e:
        print(f"\n[Inference Error] {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
