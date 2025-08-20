Make sure required libraries found at requirements.txt are installed before running application
three main commands are found within docFinder --ingest, --purge, and --query, more info can be currently found in main.py, ex queries below

python3 main.py --ingest --mode semantic  --dir Documents 
python3 main.py --query "examplePrompt" --mode semantic  --dir Documents 
python3 main.py --purge --mode semantic  --dir Documents 
