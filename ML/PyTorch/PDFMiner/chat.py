import argparse

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from text_generation import Client



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=100)
    return parser.parse_args()


def embed(fname, window_size, step_size):
    text = extract_text(fname)
    text = " ".join(text.split())
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), step_size):
        window = text_tokens[i : i + window_size]
        if len(window) < window_size:
            break
        sentences.append(window)

    paragraphs = [" ".join(s) for s in sentences]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.max_seq_length = 512
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    embeddings = model.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return model, cross_encoder, embeddings, paragraphs


if __name__ == "__main__":
    args = parse_args()
    
    model, cross_encoder, embeddings, paragraphs = embed(
        args.fname,
        args.window_size,
        args.step_size,
    )
    print(embeddings.shape)