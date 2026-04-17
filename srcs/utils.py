def get_best_token(authorized: set[int], logits: list[float]):
    return max(authorized, key=lambda i: logits[i])
