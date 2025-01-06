import argparse

from utils import load_data, save_data

def preprocess(input_path, output_path):
    # Implementation of preprocessing logic
    data,target = load_data(input_path)

    # Preprocess data
    # Get 3 random columns from data and sum with target column
    x = data[:, random.sample(range(data.shape[1]), 3)] + target

    # Save preprocessed data
    save_data(x, output_path)
    
    print(f"Preprocessing {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    args = parser.parse_args()

    preprocess(args.input, args.output)
