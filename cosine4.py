# ---- Preprocessing ----
def preprocess(text):
    text = text.lower()
    for ch in [",", ".", "!", "?", ";", ":"]:
        text = text.replace(ch, "")
    return text.split()

# ---- Vocabulary ----
def build_vocab(tokens1, tokens2):
    vocab = list(set(tokens1 + tokens2))  # unique words
    vocab.sort()
    return vocab

# ---- Bag of Words ----
def vectorize(tokens, vocab):
    return [tokens.count(word) for word in vocab]

# ---- Cosine Similarity ----
def cosine_similarity(vec1, vec2, vocab):
    dot = 0
    print("\n--- Cosine Similarity Calculation ---")
    print("Word".ljust(15), "A".rjust(3), "B".rjust(3), "A*B".rjust(5))
    print("-" * 40)
    for word, a, b in zip(vocab, vec1, vec2):
        product = a * b
        print(word.ljust(15), str(a).rjust(3), str(b).rjust(3), str(product).rjust(5))
        dot += product
    norm1 = sum([a * a for a in vec1]) ** 0.5
    norm2 = sum([b * b for b in vec2]) ** 0.5
    print("\nDot Product =", dot)
    print("||A|| =", round(norm1, 3), " ||B|| =", round(norm2, 3))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ---- Jaccard Similarity ----
def jaccard_similarity(set1, set2):
    inter = set1.intersection(set2)
    union = set1.union(set2)
    return len(inter) / len(union) if union else 0.0

# ---- MAIN PROGRAM ----
doc1 = "Artificial intelligence and machine learning are transforming healthcare by enabling early diagnosis and personalized treatment."
doc2 = "Machine learning techniques are widely applied in healthcare to support early disease detection, medical imaging, and treatment recommendations."

# Preprocess
tokens1 = preprocess(doc1)
tokens2 = preprocess(doc2)

# Vocabulary
vocab = build_vocab(tokens1, tokens2)

# Vectors
vec1 = vectorize(tokens1, vocab)
vec2 = vectorize(tokens2, vocab)

# Cosine Similarity
cos_sim = cosine_similarity(vec1, vec2, vocab)
print("Cosine Similarity:", round(cos_sim, 3))

# Jaccard Similarity
set1 = set(tokens1)
set2 = set(tokens2)
jac_sim = jaccard_similarity(set1, set2)
print("Jaccard Similarity:", round(jac_sim, 3))
