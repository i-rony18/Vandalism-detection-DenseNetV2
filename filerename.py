import os

def addsuffix_to_files(folder_path, suffix):
    """
    Add a suffix to the file names in a folder.

    Args:
        folder_path (str): Path to the folder containing the files.
        suffix (str): Suffix to be added to the file names.

    """
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            name, extension = os.path.splitext(filename)
            new_name = f"{name}{suffix}{extension}"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_name}'.")
            except FileNotFoundError:
                print(f"File '{filename}' not found.")

addsuffix_to_files("C:/Users/Ronit Das/Desktop/NSUT work/Video Analytics/Vandalism detection/other dataset/graffiti4", "g5")