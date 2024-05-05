import json
from argparse import ArgumentParser
from pathlib import Path

from rdkit import Chem


def main():
    parser = ArgumentParser(description="Generate star map as a JSON file")
    parser.add_argument("sdf_file", help="Path to sdf file containing mols")
    parser.add_argument(
        "hub_compound_name", help="Compound to be at the center of the star map"
    )
    parser.add_argument(
        "--output-path",
        help="Path to write out, else will be sdf_file filename without extension with '_edges.json' appended",
    )
    args = parser.parse_args()

    sdf_path = Path(args.sdf_file).expanduser()
    mols = [mol for mol in Chem.SDMolSupplier(str(sdf_path))]
    mols_by_name = {mol.GetProp("_Name"): mol for mol in mols}
    assert len(mols) == len(
        mols_by_name
    ), "Mol names are not unique, unable to generate star map"

    assert (
        args.hub_compound_name in mols_by_name
    ), f"Unable to find hub compound: {args.hub_compound_name}"

    output_path = args.output_path
    if output_path is None:
        output_path = f"{sdf_path.name}_edges.json"
    edges = []
    for mol_name in mols_by_name.keys():
        if mol_name == args.hub_compound_name:
            continue
        edges.append((args.hub_compound_name, mol_name))
    print(f"Writing star map edges to {output_path}")
    with open(output_path, "w") as ofs:
        json.dump(edges, ofs)


if __name__ == "__main__":
    main()
