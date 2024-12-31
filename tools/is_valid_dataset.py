import os
from tqdm import tqdm

def check_labels(labels_dir):
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    invalid_files = []
    total_files = 0
    total_invalid = 0
    
    # Get all txt files
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"Checking {len(txt_files)} label files...")
    
    # Process each file
    for filename in tqdm(txt_files, desc="Checking labels"):
        filepath = os.path.join(labels_dir, filename)
        total_files += 1
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # get yolo id
                        class_id = int(float(line.strip().split()[0]))
                        
                        # Check if class_id is valid
                        if class_id < 0 or class_id > 7:
                            total_invalid += 1
                            invalid_files.append({
                                'file': filename,
                                'line': line_num,
                                'class_id': class_id
                            })
                            
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num} in {filename}: {e}")
                        
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            
    # Save results
    with open('logs/invalid_labels.log', 'w') as f:
        f.write(f"Checked {total_files} files\n")
        f.write(f"Found {total_invalid} invalid labels\n\n")
        
        if invalid_files:
            f.write("Invalid labels found in:\n")
            for item in invalid_files:
                f.write(f"File: {item['file']}, Line: {item['line']}, Class: {item['class_id']}\n")
        else:
            f.write("No invalid labels found\n")
            
    print(f"\nResults saved to logs/invalid_labels.log")
    print(f"Total files checked: {total_files}")
    print(f"Total invalid labels: {total_invalid}")

if __name__ == "__main__":
    LABELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/labels'))
    check_labels(LABELS_DIR)