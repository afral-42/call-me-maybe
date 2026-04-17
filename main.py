# from llm_sdk import Small_LLM_Model
# from math import inf
# import json
# from trie import Trie
# from states import StaticStringState, StringRouterState, Automate


def get_token_trie(llm: Small_LLM_Model) -> Trie:
    trie = Trie()

    path = llm.get_path_to_tokenizer_file()
    with open(path, "r") as f:
        tokenizer_file = json.load(f)

    vocab = tokenizer_file["model"]["vocab"]
    for string, token in vocab.items():
        trie.insert(llm.decode(token), token)

    return trie


def get_best_token(authorized: list[int], logits: list[float]):
    return max(authorized, key=lambda i: logits[i])


def generate():
    print("Chargement du modèle...")
    llm = Small_LLM_Model(device="cpu") # Ou sans cpu si le GPU est débloqué

    # 1. On crée la discussion
    messages = [
        {"role": "system", "content": "Tu es un CTO talentueux et un développeur hors pair"},
        {"role": "user", "content": "Code moi une fonction python qui trie une chaine de caractere dans l'ordre alphabetique"}
    ]

    # 2. On utilise le tokenizer pour appliquer le format "Chat"
    # add_generation_prompt=True ajoute le petit tag <|im_start|>assistant à la fin 
    # pour forcer le modèle à commencer sa réponse !
    prompt_formate = llm._tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print(prompt_formate)
    print("---------------------------------------------\n")

    # 3. On encode ce prompt formaté
    tensor_prompt = llm.encode(prompt_formate)
    input_ids_list = tensor_prompt[0].tolist()

    # 4. Récupération des logits

    token_fin = llm._tokenizer.eos_token_id

    max_index = 0
    logits = llm.get_logits_from_input_ids(input_ids_list)

    while max_index != token_fin:
        max_index = 0
        max_logit = -inf
        for i, logit in enumerate(logits):
            if logit > max_logit:
                max_logit = logit
                max_index = i

        token_text = llm.decode([max_index])
        print(token_text, end="", flush=True)

        input_ids_list.append(max_index)
        logits = llm.get_logits_from_input_ids(input_ids_list)


def test_automate():
    # Nous voulons forcer ce rendu
    llm = Small_LLM_Model(device="cpu")  # Ou sans cpu si le GPU est débloqué

    automate = StaticStringState("{\"name\": \"")
    trie = get_token_trie(llm)

    messages = [
        {"role": "system", "content": "You are a ultra futuristic IT assistant who can interact with the world"},
        {"role": "user", "content": "Response ONLY a structured json of a function to add two numbers"}
    ]

    prompt_formate = llm._tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 3. On encode ce prompt formaté
    tensor_prompt = llm.encode(prompt_formate)
    input_ids_list = tensor_prompt[0].tolist()

    while not automate.is_done():
        authorized = set(automate.get_valid_tokens(trie))
        logits = llm.get_logits_from_input_ids(input_ids_list)

        for index in range(trie.size):
            if index not in authorized:
                logits[index] = -inf

        best_token = get_best_token(authorized, logits)
        automate.consume(llm.decode(best_token))
        print(llm.decode(best_token), end="", flush=True)
        input_ids_list.append(best_token)


def get_string_router_automate(options: list[str]) -> StringRouterState:
    return StringRouterState({StaticStringState(option) for option in options})


def test_automate_choose_function():
    # Nous voulons forcer ce rendu
    llm = Small_LLM_Model(device="cpu")  # Ou sans cpu si le GPU est débloqué

    automate = get_string_router_automate([
        "fn_greet_user",
        "fn_add_numbers",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_send_money"
    ])
    trie = get_token_trie(llm)

    funcs = "fn_greet_user,fn_add_numbers,fn_reverse_string,fn_get_square_root,fn_send_money"
    messages = [
        {"role": "system", "content": "You are a ultra futuristic IT assistant who can interact with the world"},
        {"role": "user", "content": f"Please, choose only one function to make a payment in the following set: {funcs}, you must return a valid json with parameters"}
    ]

    prompt_formate = llm._tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 3. On encode ce prompt formaté
    tensor_prompt = llm.encode(prompt_formate)
    input_ids_list = tensor_prompt[0].tolist()

    while not automate.is_done():
        authorized = set(automate.get_valid_tokens(trie))
        logits = llm.get_logits_from_input_ids(input_ids_list)

        for index in range(len(logits)):
            if index not in authorized:
                logits[index] = -inf

        best_token = get_best_token(logits, trie.size)
        automate.consume(llm.decode(best_token))
        print(llm.decode(best_token), end="", flush=True)
        input_ids_list.append(best_token)


from srcs.inference import InferenceEngine
import time

if __name__ == "__main__":
    begin = time.perf_counter()
    engine = InferenceEngine()
    engine.run()
    end = time.perf_counter()
    print(end - begin)
