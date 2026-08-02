from tokenizers import Tokenizer

tok = Tokenizer.from_file("models/qwen3-4b-base/tokenizer.json")
with open("reference/corpus.txt", newline="") as f:
    for line in f:
        print(tok.encode(line, add_special_tokens=False).ids)
