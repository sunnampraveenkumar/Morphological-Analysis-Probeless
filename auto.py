import os
import pickle
import subprocess
import shutil
import argparse
from pathlib import Path

# Constants
MODEL = 'bert'
LANGUAGE = 'eng'
ATTRIBUTE = 'Part of Speech'
LAYER = 8
RANKING = 'ttb linear'
RANKING_1 = 'ttb probeless'

# Paths (Update these paths with your actual data paths)
data_base_dir = "/raid/home/dgx1405/group3/Individual-Neurons-Pitfalls-main/EngEWT_splited_data"
pickle_dir = "/raid/home/dgx1405/group3/probeless_codes-temp1/pickles"
results_dir = "/raid/home/dgx1405/group3/probeless_codes-temp1/results"

def get_batch_paths(data_base_dir, num_batches, batch_index):
    batch_dir = f"100_{num_batches}/en{batch_index}" 
    train_path = Path(data_base_dir) / batch_dir / "en_ewt-um-train.conllu"
    dev_path = Path(data_base_dir) / batch_dir / "en_ewt-um-dev.conllu"
    test_path = Path(data_base_dir) / batch_dir / "en_ewt-um-test.conllu"
    return train_path, dev_path, test_path

def delete_directories(*dirs):
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted directory: {dir_path}")
        else:
            print(f"Directory not found: {dir_path}")

def main():
    for num_batches in [20]:  # Outer loop for num_batches = 100, 200, 400
        print(f"Processing with num_batches = {num_batches}")
        
        for i in range(num_batches):  # Inner loop for each batch
            try:
                subprocess.run(
                    ["python", "parsing.py", "-model", MODEL, "-language", LANGUAGE, 
                     "-num_batches", str(num_batches), "-batch", str(i)], 
                    check=True
                )

                subprocess.run(
                    ["python", "LinearWholeVector.py", "-model", MODEL, "-language", LANGUAGE, 
                     "-attribute", ATTRIBUTE, "-layer", str(LAYER)], 
                    check=True
                )

                att_path = Path(pickle_dir) / 'UM' / MODEL / LANGUAGE / ATTRIBUTE
                values_to_ignore_path = att_path / 'values_to_ignore.pkl'

                if values_to_ignore_path.exists():
                    with open(values_to_ignore_path, 'rb') as f:
                        values_to_ignore = pickle.load(f)

                    if values_to_ignore:
                        print(f"Batch {i} contains values to ignore. Deleting directories and moving to the next batch.")
                        # delete_directories(pickle_dir, results_dir)
                        continue

                subprocess.run(
                    ["python", "Probeless.py", "-model", MODEL, "-language", LANGUAGE, 
                     "-attribute", ATTRIBUTE, "-layer", str(LAYER)], 
                    check=True
                )

                subprocess.run(
                    ["python", "LinearSubset.py", "-model", MODEL, "-language", LANGUAGE, 
                     "-attribute", ATTRIBUTE, "-layer", str(LAYER), "-ranking", RANKING], 
                    check=True
                )

                subprocess.run(
                    ["python", "LinearSubset.py", "-model", MODEL, "-language", LANGUAGE, 
                     "-attribute", ATTRIBUTE, "-layer", str(LAYER), "-ranking", RANKING_1], 
                    check=True
                )

                subprocess.run(
                    ["python", "analysis.py", "-experiments", "probing", "-model", MODEL, "-language", LANGUAGE, 
                    "-attribute", ATTRIBUTE, "-layer", str(LAYER)], 
                    check=True
                )

                results_path = Path(results_dir) / f"en_{num_batches}"
                results_path.mkdir(parents=True, exist_ok=True) 

                pos_dir = Path(results_dir) / "Parts of Speech"
                if pos_dir.exists() and pos_dir.is_dir():  
                    shutil.move(str(pos_dir), str(results_path))  

                print(f"Batch {i} for num_batches = {num_batches} completed successfully.")
                break  # This `break` will stop only the inner loop

            except subprocess.CalledProcessError as e:
                print(f"An error occurred in batch {i} for num_batches = {num_batches}: {e}")
                # delete_directories(pickle_dir, results_dir)

        print(f"All batches processed for num_batches = {num_batches}.")  # End of inner loop for a specific num_batches value

    print("All num_batches sets processed.")  # End of outer loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deep learning experiments with batch processing.")
    args = parser.parse_args()
    
    main()