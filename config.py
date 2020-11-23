import os

ROOT_DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = "/Users/briancai/Desktop/Datasets/yelp_dataset/yelp_academic_dataset_review.json"
OUTPUT_PATH = os.path.join(ROOT_DIRECTORY_PATH, "output")
MODEL_PATH = os.path.join(OUTPUT_PATH, "final_model")
