import argparse
from src.search import SearchIndex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train_full")
    args = parser.parse_args()
    
    print(f"Rebuilding FAISS index for split: {args.split}...")
    idx = SearchIndex()
    idx.build_from_cache()
    print("Done.")

if __name__ == "__main__":
    main()
