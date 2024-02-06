import os

def write_file_start(start_time, file_path, with_header=True):
    with open(file_path, "a") as file:
        # Check if file is empty to avoid adding an extra newline at the beginning
        if os.stat(file_path).st_size == 0:
            file.write(f"Start Experiment {start_time}\n")
        else:
            file.write(f"\nStart Experiment {start_time}")
        if with_header:
            file.write(f"\nExperiment, validity, proximity, validity_without_chem, proximity_without_chem")
