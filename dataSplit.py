# import os

# def split_conllu_file(src_file, num_splits, dest_dirs):
#     # Read the content of the source file
#     with open(src_file, 'r', encoding='utf-8') as file:
#         data = file.read()
    
#     # Split the data into sentences using double newline as delimiter
#     sentences = data.strip().split('\n\n')
#     print("sent_size", len(sentences))
    
#     # Calculate the split sizes
#     split_size = len(sentences) // num_splits
#     print("split_sizes", split_size)
    
    
#     # Ensure each directory exists
#     for dest_dir in dest_dirs:
#         os.makedirs(dest_dir, exist_ok=True)
    
#     # Split sentences and write to destination files
#     for i in range(num_splits):
#         start_index = i * split_size
#         if i == num_splits - 1:
#             end_index = len(sentences)
#         else:
#             end_index = (i + 1) * split_size

#         subset_sentences = sentences[start_index:end_index]
#         dest_file = os.path.join(dest_dirs[i], os.path.basename(src_file))
        
#         # Write the subset of sentences to the file, joining with double newline
#         with open(dest_file, 'w', encoding='utf-8') as file:
#             file.write('\n\n'.join(subset_sentences) + '\n\n')

# def process_conllu_file(src_file, splits_info):
#     # Process the Conllu file for different splitting configurations
#     for num_splits, dest_dirs in splits_info:
#         split_conllu_file(src_file, num_splits, dest_dirs)

# def main():
#     # Path to the source Conllu file
#     src_file = '/home/ws1405/tripti/probeless_codes/data/UM/fr/fr_sequoia-um-train.conllu'

#     # Define the splitting configurations
#     splits_info = [
#         (2, ['/home/ws1405/tripti/probeless_codes/splited_data/100_2/fr' + str(i) for i in range(2)]),
#         (4, ['/home/ws1405/tripti/probeless_codes/splited_data/100_4/fr' + str(i) for i in range(4)]),
#         (10, ['/home/ws1405/tripti/probeless_codes/splited_data/100_10/fr' + str(i) for i in range(10)]),
#         (20, ['/home/ws1405/tripti/probeless_codes/splited_data/100_20/fr' + str(i) for i in range(20)]),
#         (50, ['/home/ws1405/tripti/probeless_codes/splited_data/100_50/fr' + str(i) for i in range(50)]),
#     ]

#     # Process the Conllu file
#     process_conllu_file(src_file, splits_info)

# if __name__ == "__main__":
#     main()


###########################################################################################

import os

language_map = {
    'eng': 'en',
    'chi': 'zh',
    'hin': 'hi',
    'heb': 'he',
    'jap': 'ja'
}
def split_conllu_file(src_file, num_splits, dest_dirs):
    # Read the content of the source file
    with open(src_file, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Split the data into sentences using double newline as delimiter
    sentences = data.strip().split('\n\n')
    
    # Calculate the split sizes
    split_size = len(sentences) // num_splits
    
    # Ensure each directory exists
    for dest_dir in dest_dirs:
        os.makedirs(dest_dir, exist_ok=True)
    
    # Split sentences and write to destination files
    for i in range(num_splits):
        start_index = i * split_size
        if i == num_splits - 1:
            end_index = len(sentences)
        else:
            end_index = (i + 1) * split_size

        subset_sentences = sentences[start_index:end_index]
        dest_file = os.path.join(dest_dirs[i], os.path.basename(src_file))
        
        # Write the subset of sentences to the file, joining with double newline
        with open(dest_file, 'w', encoding='utf-8') as file:
            file.write('\n\n'.join(subset_sentences) + '\n\n')
        
        # Print the length of the file after splitting
        print(f"Split {i + 1}: {len(subset_sentences)} sentences written to {dest_file}")

def process_conllu_file(src_file, lang):
    # Process the Conllu file for different splitting configurations
    splits_info = [
        (2, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_2/{language_map[lang]}' + str(i) for i in range(2)]),  # 50%
        (4, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_4/{language_map[lang]}' + str(i) for i in range(4)]),  # 25%
        (10, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_10/{language_map[lang]}' + str(i) for i in range(10)]),  # 10%
        (20, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_20/{language_map[lang]}' + str(i) for i in range(20)]),  # 5%
        (50, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_50/{language_map[lang]}' + str(i) for i in range(50)]),  # 2%
        (100, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_100/{language_map[lang]}' + str(i) for i in range(100)]),  # 1%
        (200, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_200/{language_map[lang]}' + str(i) for i in range(200)]),  # 0.5%
        (400, [f'/raid/home/dgx1405/group3/probeless_codes-temp1/splited_data/100_400/{language_map[lang]}' + str(i) for i in range(400)]),  # 0.25%
    ]
    for num_splits, dest_dirs in splits_info:
        split_conllu_file(src_file, num_splits, dest_dirs)

def process_all_files(train_paths):
    for lang, src_file in train_paths.items():
        print(f"Processing {lang} file:")
        process_conllu_file(src_file, lang)

def main():
    # Define the splitting configurations
    

    # Define train_paths dictionary
    train_paths = {
        'eng': '/raid/home/dgx1405/group3/probeless_codes-temp1/data/UM/en/en_ewt-um-train.conllu',
        'chi': '/raid/home/dgx1405/group3/probeless_codes-temp1/data/UM/zh/zh_gsd-um-train.conllu',
        'hin': '/raid/home/dgx1405/group3/probeless_codes-temp1/data/UM/hi/hi_hdtb-um-train.conllu',
        'heb': '/raid/home/dgx1405/group3/probeless_codes-temp1/data/UM/he/he_htb-um-train.conllu',
        'jap': '/raid/home/dgx1405/group3/probeless_codes-temp1/data/UM/ja/ja_gsd-um-train.conllu'
    }

    # Process all files
    process_all_files(train_paths)

if __name__ == "__main__":
    main()