'''
# Script Description: CIF to PDB Batch Converter
This Python script converts multiple macromolecular structure files from mmCIF format (.cif) to PDB format (.pdb) using the Biopython Bio.PDB module. The script scans a specified directory containing .cif files, parses each structure, and writes the equivalent structure to a new .pdb file in an output directory.
The conversion is useful in structural bioinformatics workflows, where many downstream tools (structural alignment, docking, molecular visualization, or fold analysis) still require PDB format instead of mmCIF.

# The script automatically:
1. Reads all .cif files from an input directory.
2. Parses each structure using MMCIFParser.
3. Converts and saves the structure in .pdb format.
4. Stores the converted files in a designated output directory.
5. Creates the output directory if it does not already exist.
6. This enables efficient batch conversion of large structural datasets without manual file-by-file processing.

# Directory Structure
project_folder/
│
├── cif_files/
│   ├── structure1.cif
│   ├── structure2.cif
│   └── structure3.cif
│
├── pdb_files/
│
└── cif_to_pdb_converter.py

cif_files/ → contains the input .cif files.
pdb_files/ → output directory where converted .pdb files will be saved.

# How to Run the Script
Run the script:

python cif_to_pdb_converter.py

After execution, the converted .pdb files will appear in the pdb_files directory.

# Example output:

Converted: structure1.pdb
Converted: structure2.pdb
Converted: structure3.pdb
All conversions complete.
Example Use Case

This script is particularly useful for workflows involving:

Protein structural comparison

Fold analysis

Molecular docking

Visualization in tools that prefer .pdb

'''

import os
from Bio.PDB import MMCIFParser, PDBIO

input_dir = "cif_files"
output_dir = "pdb_files"

os.makedirs(output_dir, exist_ok=True)

parser = MMCIFParser(QUIET=True)

for file in os.listdir(input_dir):

    if file.endswith(".cif"):

        cif_path = os.path.join(input_dir, file)

        structure = parser.get_structure(file, cif_path)

        pdb_file = file.replace(".cif", ".pdb")
        pdb_path = os.path.join(output_dir, pdb_file)

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)

        print("Converted:", pdb_file)

print("All conversions complete.")