import os
import re
import pandas as pd

# Define the base paths
base_path_ttb_linear = "/raid/home/dgx1405/group3/probeless_codes-temp1/results/en_{batch}/UM/bert/eng/Part of Speech/layer 8/linear by ttb linear"
base_path_ttb_probeless = "/raid/home/dgx1405/group3/probeless_codes-temp1/results/en_{batch}/UM/bert/eng/Part of Speech/layer 8/linear by ttb probeless"

# Define the output path
output_base_path = "/raid/home/dgx1405/group3/probeless_codes-temp1/results_xl"

# Create output base path if it doesn't exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Define the batches
num_batches = [2, 4, 10, 50, 100, 200, 400]

# Initialize a dictionary to hold data for the neurons as rows
data = {}

# Helper function to process a file and add data to the dictionary
def process_file(file_path, batch, file_type):
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            input_text = file.read()

        # Extract neuron count, train accuracy, and test accuracy using regex
        neurons = re.findall(r'using (\d+) neurons', input_text)
        train_accuracies = re.findall(r'accuracy on train set: ([\d.]+)', input_text)
        test_accuracies = re.findall(r'final accuracy on test: ([\d.]+)', input_text)

        # Add data to the dictionary for each neuron
        for neuron, train_acc, test_acc in zip(neurons, train_accuracies, test_accuracies):
            neuron = int(neuron)  # Convert neuron to int
            # Initialize the row if it doesn't exist
            if neuron not in data:
                data[neuron] = {}
            
            # Add the train and test accuracies to the correct columns for this neuron
            data[neuron][f'batch_{batch}_{file_type}_train'] = float(train_acc)
            data[neuron][f'batch_{batch}_{file_type}_test'] = float(test_acc)

        print(f"Data processed from {file_path} for batch {batch} ({file_type})")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Loop through each batch and process both ttb linear and probeless files
for batch in num_batches:
    # Construct the file paths for both linear and probeless files
    linear_file_path = base_path_ttb_linear.format(batch=batch)
    probeless_file_path = base_path_ttb_probeless.format(batch=batch)

    # Process the linear file
    if os.path.exists(linear_file_path):
        process_file(linear_file_path, batch, "linear")
    else:
        print(f"Linear file not found for batch {batch}: {linear_file_path}")

    # Process the probeless file
    if os.path.exists(probeless_file_path):
        process_file(probeless_file_path, batch, "probeless")
    else:
        print(f"Probeless file not found for batch {batch}: {probeless_file_path}")

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Define the output path for the single Excel file
output_file = os.path.join(output_base_path, "all_batches_data_combined.xlsx")

# Save the DataFrame to Excel
df.to_excel(output_file, index_label='Neurons')
print(f"All data successfully saved to {output_file}")
