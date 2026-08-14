"""
Create a comprehensive zip archive containing PEFT benchmark results (excluding checkpoints) and generated visualization plots.
"""

import os
import zipfile

def create_results_archive(results_dir='results', plots_dir='plots', output_zip='peft_benchmark_results.zip'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, results_dir)
    plots_dir = os.path.join(base_dir, plots_dir)
    output_zip = os.path.join(base_dir, output_zip)
    
    print(f"Creating archive: {output_zip}")
    print(f"Adding from results: {results_dir} (excluding checkpoints)")
    print(f"Adding from plots:   {plots_dir}\n")
    
    file_count = 0
    total_uncompressed_bytes = 0
    
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Add results directory (excluding checkpoints)
        for root, dirs, files in os.walk(results_dir):
            dirs[:] = [d for d in dirs if not d.startswith('checkpoint')]
            parts = os.path.relpath(root, base_dir).split(os.sep)
            if any(p.startswith('checkpoint') for p in parts):
                continue
                
            for file in sorted(files):
                if file.endswith('.safetensors') or file.endswith('.bin'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
                file_count += 1
                total_uncompressed_bytes += os.path.getsize(file_path)
                
        # 2. Add plots directory
        if os.path.exists(plots_dir):
            for root, dirs, files in os.walk(plots_dir):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, base_dir)
                    zipf.write(file_path, arcname)
                    file_count += 1
                    total_uncompressed_bytes += os.path.getsize(file_path)
                    
    zip_size = os.path.getsize(output_zip)
    print("=" * 50)
    print(f"Archive Created Successfully!")
    print(f"Total Files Archived:      {file_count}")
    print(f"Uncompressed Size:         {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"Compressed Zip Size:       {zip_size / (1024 * 1024):.2f} MB ({zip_size:,} bytes)")
    print(f"Saved to:                  {output_zip}")
    print("=" * 50)

if __name__ == '__main__':
    create_results_archive()
