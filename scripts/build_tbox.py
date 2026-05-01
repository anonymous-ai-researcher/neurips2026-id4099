"""Extract TBox axioms from Cell Ontology OWL file."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build TBox from CL OWL")
    parser.add_argument("--cl-owl", type=str, required=True)
    parser.add_argument("--uberon-owl", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # TODO: Parse OWL with owlready2, extract EL-bot axioms
    print(f"TBox saved to {args.output}")

if __name__ == "__main__":
    main()
