from srcs.inference import InferenceEngine
import time


if __name__ == "__main__":
    begin = time.perf_counter()
    engine = InferenceEngine()
    engine.run()
    engine.save_answer()  # AJouter a l'init un path de sauvegarde, un path de recup
    end = time.perf_counter()
    print(end - begin)
