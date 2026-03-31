from collection import Counter

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i],symbols[i+1])] += freq
    return pairs

def merge_vocab(pair,vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement ="".join(pair)

    for word in vocab:
        new_word = word.replace(bigram,replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

def train_bpe(corpus,num_merges=10):
    vocab = Counter()
    for word in corpus:
        chars = " ".join(list(word)) + " </w>"
        vocab[chars] += 1

    merges = []

    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs,key=pairs.get)
        merges.append(best)

def encode(word, merges):
    tokens = list(word) + ["</w>"]

    for pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                tokens[i:i+2] = ["".join(pair)]
            else:
                i += 1

    return tokens

corpus = ["low", "lower", "newest", "widest"]
merges = train_bpe(corpus, num_merges=10)
print("merges:", merges)
print(encode("lowest", merges))
