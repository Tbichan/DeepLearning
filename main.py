import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def main():

    # training fast
    cudnn.benchmark = True

    # Directories.
    log_dir = "log"
    sample_dir = "sample"
    model_save_dir = "model"
    result_dir = "resule"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    myDataset = get_loader('dataset/hair', crop_size=178, image_size=128,
               batch_size=16, mode="train", num_workers=1)

    solver = Solver(myDataset)
    solver.train()

if __name__ == "__main__":
    main()
