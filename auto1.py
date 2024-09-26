import os
import pickle
import subprocess
import shutil
import argparse
from pathlib import Path
from consts import language_map

# Constants
MODEL = 'bert'
ATTRIBUTE = 'Part of Speech'
LAYER = 8
RANKING = 'ttb linear'
RANKING_1 = 'ttb probeless'

data_base_dir = "/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data"
pickle_dir = "/raid/home/dgx1405/group3/probeless_codes-temp1/pickles"
results_dir = "/raid/home/dgx1405/group3/probeless_codes-temp1/results"

# def get_batch_paths(data_base_dir, num_batches, batch_index):
#     batch_dir = f"100_{num_batches}/en{batch_index}" 
#     train_path = Path(data_base_dir) / batch_dir / "en_ewt-um-train.conllu"
#     dev_path = Path(data_base_dir) / batch_dir / "en_ewt-um-dev.conllu"
#     test_path = Path(data_base_dir) / batch_dir / "en_ewt-um-test.conllu"
#     return train_path, dev_path, test_path

def delete_directories(language, *dirs):
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            for item in path.iterdir():
                # Check if item is a file and does not match the naming format
                if item.is_file() and not item.name.startswith(f"{language_map[language]}_"):
                    item.unlink()  # Delete the file

                # Check if item is a directory and does not match the naming format
                elif item.is_dir() and not item.name.startswith(f"{language_map[language]}_"):
                    # Recursively delete the directory and its contents
                    for dir_path in item.iterdir():
                        if os.path.exists(dir_path):
                            shutil.rmtree(dir_path)
                            print(f"Deleted directory: {dir_path}")
                    item.rmdir()

def move_non_matching_files(language, src_dir, dest_dir, num_batches):
    for item in Path(src_dir).iterdir():
        if item.is_file() and not item.name.startswith(f"{language_map[language]}_"):
            shutil.move(str(item), str(dest_dir))


def main():
    for num_batches in [2,4,10,20,100,200,400]: 
        print(f"Processing with num_batches = {num_batches}")
        
        for i in range(num_batches):  
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
                        delete_directories(LANGUAGE, pickle_dir, results_dir)
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

                results_path = Path(results_dir) / f"{language_map[LANGUAGE]}_{num_batches}"
                results_path.mkdir(parents=True, exist_ok=True)  

                pos_dir = Path(results_dir) / "UM"  
                if pos_dir.exists() and pos_dir.is_dir(): 
                    shutil.move(str(pos_dir), str(results_path))

                pickle_dest_path = Path(pickle_dir) / f"{language_map[LANGUAGE]}_{num_batches}"
                pickle_dest_path.mkdir(parents=True, exist_ok=True)  

                move_non_matching_files(LANGUAGE, pickle_dir, pickle_dest_path, num_batches)

                print(f"Batch {i} for num_batches = {num_batches} completed successfully.")
                break

            except subprocess.CalledProcessError as e:
                print(f"An error occurred in batch {i} for num_batches = {num_batches}: {e}")
                delete_directories(LANGUAGE, pickle_dir, results_dir)

        print(f"All batches processed for num_batches = {num_batches}.")  

    print("All num_batches sets processed.") 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deep learning experiments with batch processing.")
    parser.add_argument('--language', type=str, help='Specify the language to run experiments in.')
    args = parser.parse_args()
    LANGUAGE = args.language
    main()