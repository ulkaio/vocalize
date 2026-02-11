import os
import argparse


def count_files(directory=".", recursive=False):
    """Count files in a directory.
    
    Args:
        directory: Path to directory to count files in (default: current directory)
        recursive: If True, count files in subdirectories too
    
    Returns:
        int: Number of files
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory")
        return 0
    
    count = 0
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            count += len(files)
    else:
        for entry in os.scandir(directory):
            if entry.is_file():
                count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description="Count files in a directory")
    parser.add_argument("directory", nargs="?", default=".", 
                        help="Directory path (default: current directory)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Count files recursively in subdirectories")
    
    args = parser.parse_args()
    
    file_count = count_files(args.directory, args.recursive)
    print(f"Files found: {file_count}")


if __name__ == "__main__":
    main()
