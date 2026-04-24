from srcs.inference import BatchInferenceEngine
import time


def main() -> None:
    begin = time.perf_counter()
    engine = BatchInferenceEngine.build_engine(
        input_path="data/input/function_calling_tests.json",
        functions_definition_path="data/input/functions_definition.json",
        output_path="result.json"
    )
    engine.run()
    engine.save_results()
    end = time.perf_counter()
    print(end - begin)


if __name__ == "__main__":
    main()
