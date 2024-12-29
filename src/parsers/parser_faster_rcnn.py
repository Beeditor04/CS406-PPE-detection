import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Parser for Faster R-CNN training")
    parser.add_argument("--data_dir", type=str, default="data/split", help="Path to dataset folder")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Frequency of evaluation")
    return parser

def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args