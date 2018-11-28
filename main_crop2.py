import os
import argparse
from solver import Solver
from data_loader import get_loader_crop2resize
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

    myDataset = get_loader_crop2resize('dataset/hair', crop_size=178, image_size=128,
               batch_size=16, mode="train", num_workers=1)

    solver = Solver(myDataset)
    solver.num_iters = 400000
    solver.num_iters_decay = 200000
    solver.train(240000)

if __name__ == "__main__":
    main()
