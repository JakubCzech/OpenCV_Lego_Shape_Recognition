import argparse
from pathlib import Path
from processing.utils import Project

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    project_instance = Project(images_dir)
    project_instance.save_result(results_file)

# def main_for_testing():
#     import time 
#     print("Start")
#     start = time.time()
#     project_instance = Project(Path("train"))
#     project_instance.save_result(Path("results.json"))  
#     print(time.time() - start)

if __name__ == '__main__':
    main()
    # main_for_testing()