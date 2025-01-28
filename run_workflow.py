#!/usr/bin/env python3

import os
import shutil
import subprocess
import re
import time
from datetime import datetime

import pandas as pd


import generator.MoleculeBuilder as Mb
import generator.ChainBuilder as ChB


class DFTBCalculator:
    """
    A simplified version of your DFTBCalculator to handle .xyz files.
    If you also want to handle .vasp for periodic cases, you can adapt
    the 'Geometry' block or keep a separate logic.
    """
    def __init__(self, xyz_file, work_dir="dftb_calc"):
        self.xyz_file = xyz_file
        self.work_dir = work_dir
        self.template = """Geometry = XYZFormat {
  <<< "geometry.xyz"
}

Driver = {}

Hamiltonian = DFTB {
  SCC = Yes
  SCCTolerance = 1e-6
  charge = %d  # Placeholder for charge
  MaxAngularMomentum = {
%s
}

SlaterKosterFiles = Type2FileNames {
    Separator = "-"
    Suffix = ".skf"
    Prefix = "3ob/"
}

Filling = Fermi {
    Temperature [Kelvin] = 1000.0
}

Options {
    WriteDetailedXml = Yes
    WriteDetailedOut = Yes
}

Analysis {
    WriteBandOut = Yes
    WriteEigenvectors = Yes
}

ParserOptions {
  ParserVersion = 6
}"""

    def get_atom_types(self):
        """Parse .xyz and collect unique atom types."""
        atom_types = set()
        with open(self.xyz_file, 'r') as f:
            next(f)  # skip line with number of atoms
            next(f)  # skip comment line
            for line in f:
                if line.strip():
                    atom_type = line.split()[0]
                    atom_types.add(atom_type)
        return atom_types

    def create_angular_momentum_block(self):
        """
        Map each atom type to its max angular momentum. 
        You can extend the dictionary as needed.
        """
        momentum_map = {
            'H': 's',
            'C': 'p',
            'N': 'p',
            'O': 'p',
            'F': 'p',
            'S': 'p'
        }
        
        block = ""
        for atom in sorted(self.get_atom_types()):
            if atom in momentum_map:
                block += f"    {atom} = \"{momentum_map[atom]}\"\n"
            else:
                print(f"Warning: No defined angular momentum for {atom}, using 'p'.")
                block += f"    {atom} = \"p\"\n"
        return block

    def run_calculation(self, charge=0):
        """Run DFTB+ for a given charge, returning parsed results or None."""
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

        # Copy xyz to geometry.xyz inside the working directory
        geom_path = os.path.join(self.work_dir, "geometry.xyz")
        shutil.copy(self.xyz_file, geom_path)

        # Generate dftb_in.hsd
        angular_momentum_block = self.create_angular_momentum_block()
        dftb_in = self.template % (charge, angular_momentum_block)
        with open(os.path.join(self.work_dir, "dftb_in.hsd"), "w") as f:
            f.write(dftb_in)

        # Run DFTB+ (ensure dftb+ is in PATH)
        try:
            subprocess.run(["dftb+"], cwd=self.work_dir, check=True)
            # Parse results
            return self.parse_single_calculation()
        except subprocess.CalledProcessError:
            print(f"Error running DFTB+ calculation for charge {charge}")
            return None

    def parse_single_calculation(self):
        """Extract HOMO, LUMO, bandgap, Fermi level, total energy, etc."""
        results = {}
        band_file = os.path.join(self.work_dir, "band.out")
        detailed_file = os.path.join(self.work_dir, "detailed.out")

        # Parse band.out
        orbital_energies = []
        occupations = []
        if os.path.exists(band_file):
            with open(band_file, "r") as bf:
                for line in bf:
                    if ("KPT" in line) or (line.strip() == ""):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        orbital_energies.append(float(parts[1]))
                        occupations.append(float(parts[2]))

            # Find HOMO and LUMO from occupations
            for i in range(len(occupations)-1, -1, -1):
                if occupations[i] > 1.9:  # Typically ~2.0 for spin-restricted
                    results['HOMO'] = orbital_energies[i]
                    if i+1 < len(orbital_energies):
                        results['LUMO'] = orbital_energies[i+1]
                        results['Bandgap'] = results['LUMO'] - results['HOMO']
                    break

        # Parse detailed.out for Fermi level and total energy
        if os.path.exists(detailed_file):
            with open(detailed_file, "r") as df:
                content = df.read()
                # Fermi level
                fermi_match = re.search(
                    r"Fermi level:\s+([-+]?\d*\.\d+)\s+H\s+([-+]?\d*\.\d+)\s+eV", content
                )
                if fermi_match:
                    results['Fermi_level_eV'] = float(fermi_match.group(2))

                # Total energy in Hartree
                energy_match = re.search(r"Total energy:\s+([-+]?\d*\.\d+)\s+H", content)
                if energy_match:
                    results['Total_energy_H'] = float(energy_match.group(1))

        return results if results else None

    def calculate_all_properties(self):
        """Run neutral, cation, and anion calculations. Return summary of IP, EA, etc."""
        properties = {}
        # 1) Neutral
        print("[DFTB] Running neutral calculation ...")
        neutral = self.run_calculation(charge=0)
        if neutral:
            properties['HOMO'] = neutral.get('HOMO')
            properties['LUMO'] = neutral.get('LUMO')
            properties['Bandgap'] = neutral.get('Bandgap')
            properties['Fermi_level'] = neutral.get('Fermi_level_eV')
            properties['Neutral_energy'] = neutral.get('Total_energy_H')

        # 2) Cation
        print("[DFTB] Running cation calculation ...")
        cation = self.run_calculation(charge=1)
        if cation:
            properties['Cation_energy'] = cation.get('Total_energy_H')

        # 3) Anion
        print("[DFTB] Running anion calculation ...")
        anion = self.run_calculation(charge=-1)
        if anion:
            properties['Anion_energy'] = anion.get('Total_energy_H')

        # 4) Compute IP and EA if possible
        if all(k in properties for k in ['Neutral_energy', 'Cation_energy', 'Anion_energy']):
            hartree_to_eV = 27.211396132
            properties['IP'] = (properties['Cation_energy'] - properties['Neutral_energy']) * hartree_to_eV
            properties['EA'] = (properties['Neutral_energy'] - properties['Anion_energy']) * hartree_to_eV

        return properties



def main():
    start_datetime = datetime.now()
    print(f"Starting script at: {start_datetime}")

    # Step A: Read input CSV
    input_csv = "input.csv"  # Or pass via command line, sys.argv, etc.
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df_input = pd.read_csv(input_csv)
    
    # Step B: Separate data by type
    molecule_smiles = []
    polymer_finite_data = []
    polymer_periodic_data = []

    for index, row in df_input.iterrows():
        ID = str(row['ID'])
        smiles = row['SMILES']
        type_ = row['Type'].strip()
        length = row['Length'] if 'Length' in row else None

        if type_ == 'Molecule':
            molecule_smiles.append({'ID': ID, 'SMILES': smiles})
        
        elif type_ == 'Polymer-Finite':
            # Convert length to int if specified
            if pd.isna(length):
                print(f"Warning: No length specified for finite polymer {ID}. Skipping.")
                continue
            length = int(length)
            polymer_finite_data.append({'ID': ID, 'SMILES': smiles, 'Length': length})
        
        elif type_ == 'Polymer-Periodic':
            polymer_periodic_data.append({'ID': ID, 'SMILES': smiles})

        else:
            print(f"Warning: Unknown type '{type_}' for ID {ID}. Skipping.")
            continue

    # Step C: Build molecules
    # Using your existing psp.MoleculeBuilder code but wrapped in a simpler class:
    def build_molecules(mol_list):
        if not mol_list:
            return []
        df_mol = pd.DataFrame(mol_list)
        builder = Mb.Builder(
            Dataframe=df_mol,
            ID_col="ID",
            SMILES_col="SMILES",
            OutDir="molecules",
            Length=[1],        # or pass from each row if needed
            NumConf=1,
            Loop=False,
            IrrStruc=False
        )
        builder.Build()
        return df_mol['ID'].tolist()

    # Step D: Build finite polymers
    # You can reuse the same Mb.Builder, just pass length appropriately
    def build_finite_polymers(polymer_list):
        built_ids = []
        for poly in polymer_list:
            df_poly = pd.DataFrame([{'ID': poly['ID'], 'SMILES': poly['SMILES']}])
            builder = Mb.Builder(
                Dataframe=df_poly,
                ID_col="ID",
                SMILES_col="SMILES",
                OutDir="molecules",
                Length=[poly['Length']],  # use the specified length
                NumConf=1,
                Loop=False,
                IrrStruc=False
            )
            builder.Build()
            built_ids.append(poly['ID'])
        return built_ids

    # Step E: Build periodic polymers (infinite chain)
    def build_infinite_chains(periodic_list):
        if not periodic_list:
            return []
        df_chain = pd.DataFrame(periodic_list)
        chain_builder = ChB.Builder(
            Dataframe=df_chain,
            ID_col="ID",
            SMILES_col="SMILES",
            Length=["n"],         # Generate infinite chain
            Steps=50,            # Example: geometry steps
            Substeps=20,
            NCores=0,            # Use all available cores
            OutDir='chains'
        )
        chain_builder.BuildChain()
        return df_chain['ID'].tolist()

    # Actually build them
    molecule_ids = build_molecules(molecule_smiles)
    polymer_finite_ids = build_finite_polymers(polymer_finite_data)
    polymer_periodic_ids = build_infinite_chains(polymer_periodic_data)

    # Step F: Collect .xyz files 
    # For molecules + finite
    xyz_info = []  # list of (ID, xyz_path, SMILES)
    for ID in molecule_ids + polymer_finite_ids:
        dir_path = os.path.join("molecules", ID)
        if not os.path.isdir(dir_path):
            print(f"Warning: directory {dir_path} not found.")
            continue
        
        # Find xyz
        for f in os.listdir(dir_path):
            if f.endswith('.xyz'):
                xyz_file = os.path.join(dir_path, f)
                # Get SMILES from df_input
                row = df_input[df_input['ID'] == ID]
                if not row.empty:
                    smiles = row.iloc[0]['SMILES']
                    xyz_info.append((ID, xyz_file, smiles))
                else:
                    print(f"Warning: No SMILES found for ID={ID} in input.csv")

    # For periodic polymer (in your chain builder, it might be .xyz or .vasp; 
    # adjust accordingly if it produces VASP)
    for ID in polymer_periodic_ids:
        chain_dir = os.path.join("chains", ID)
        if not os.path.isdir(chain_dir):
            print(f"Warning: directory {chain_dir} not found.")
            continue
        
        for f in os.listdir(chain_dir):
            # Usually you might get a .vasp or .xyz. Adjust if needed:
            if f.endswith('.xyz'):
                xyz_file = os.path.join(chain_dir, f)
                row = df_input[df_input['ID'] == ID]
                if not row.empty:
                    smiles = row.iloc[0]['SMILES']
                    xyz_info.append((ID, xyz_file, smiles))
                else:
                    print(f"Warning: No SMILES found for ID={ID} in input.csv.")

    # Step G: Run DFTB+ on all .xyz files
    results_data = []  # will hold dict with: ID, SMILES, HOMO, LUMO, Bandgap, ...
    for (mol_id, xyz_file, smiles) in xyz_info:
        print(f"\nRunning DFTB for {mol_id}: {xyz_file}")
        calc = DFTBCalculator(xyz_file=xyz_file, work_dir=f"dftb_calc_{mol_id}")
        props = calc.calculate_all_properties()

        # Prepare row for CSV
        row_result = {
            'ID': mol_id,
            'SMILES': smiles,
            'HOMO (eV)': props.get('HOMO', None),
            'LUMO (eV)': props.get('LUMO', None),
            'Bandgap (eV)': props.get('Bandgap', None),
            'Fermi_level (eV)': props.get('Fermi_level', None),
            'IP (eV)': props.get('IP', None),
            'EA (eV)': props.get('EA', None)
        }
        results_data.append(row_result)

    # Step H: Write results to a CSV
    results_df = pd.DataFrame(results_data)
    results_csv = "dftb_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nDFTB+ results saved to: {results_csv}")

    end_datetime = datetime.now()
    print(f"\nScript completed at: {end_datetime}")
    total_duration = end_datetime - start_datetime
    print(f"Total wall-clock time: {total_duration}")


if __name__ == "__main__":
    main()
