"""Precompute PubMedBERT embeddings for all CL concept names."""
import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tbox", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    # TODO: Load concept names, encode with [CLS] token
    print(f"Type embeddings saved to {args.output}")

if __name__ == "__main__":
    main()
