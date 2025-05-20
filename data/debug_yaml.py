import yaml
import os
import sys

def main():
    path = "data/data.yaml"
    print(f"Loading YAML from: {os.path.abspath(path)}")
    
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        print("YAML successfully loaded")
        print(f"Keys: {list(data.keys())}")
        
        if 'names' in data:
            print(f"Names: {data['names']}")
            print(f"Type of names: {type(data['names'])}")
            if isinstance(data['names'], dict):
                print(f"Keys in names: {list(data['names'].keys())}")
        else:
            print("No 'names' key found in data")
    
    except Exception as e:
        print(f"Error loading YAML: {e}")
        print(f"Python version: {sys.version}")
        print(f"YAML library version: {yaml.__version__}")

if __name__ == "__main__":
    main()
