import os
import shutil
import argparse

def generate_condition_meshes(mesh_path, cond_path, expressions):
    """
    Copies specified files from subdirectories in `mesh_path` to `cond_path`.
    Each subdirectory in `mesh_path` has files matching the expressions.
    
    Args:
        mesh_path (str): Path to the source directory containing mesh data.
        cond_path (str): Path to the destination directory.
        expressions (list of str): List of expression subdirectories to copy.
    """
    for dir in os.listdir(mesh_path):
        source_dir = os.path.join(mesh_path, dir)
        target_dir = os.path.join(cond_path, dir)
        
        if not os.path.isdir(source_dir):
            continue
        
        os.makedirs(target_dir, exist_ok=True)
        
        for exp in expressions:
            source_file = os.path.join(source_dir, exp, f"{exp}_30.ply")
            if os.path.isfile(source_file):
                shutil.copy(source_file, target_dir)
            else:
                print(f"Warning: {source_file} not found.")

def main():
    parser = argparse.ArgumentParser(description="Copy specific mesh files from one directory to another.")
    parser.add_argument('--mesh_path', type=str, default='/mnt/diskone-second/COMA_FLAME_Aligned/',
                        help="Path to the directory containing mesh data.")
    parser.add_argument('--cond_path', type=str, default='/mnt/diskone-second/D2D/New_Conditions',
                        help="Path to the destination directory.")
    parser.add_argument('--expressions', type=str, nargs='+', 
                        default=['Happy', 'Sad', 'Fear', 'Ittitated1', 'Smile1', 'Ill', 'Pleased', 'Suspicious', 'Moody', 'Upset', 'Drunk2'],
                        help="List of expression subdirectories to copy.")
    
    args = parser.parse_args()
    
    copy_mesh_files(args.mesh_path, args.cond_path, args.expressions)

if __name__ == "__main__":
    main()
