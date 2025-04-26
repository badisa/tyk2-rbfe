import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from rdkit import Chem
from timemachine.fe.utils import read_sdf

REFERENCE_MOL_NAME = "4gih_blocker"
GLIDE_SCORE_PROPERTY = "r_i_glide_gscore"


def main():
    parser = ArgumentParser()
    args = parser.parse_args()

    parent_dir = Path(__file__).parent.parent

    src_mols_by_name = {
        mol.GetProp("_Name"): mol
        for mol in read_sdf(
            parent_dir / "data" / "10k_most_similar_charged_unique_names.sdf"
        )
    }

    results_dir = parent_dir / "results"
    assert results_dir.is_dir()

    csv_rows = []
    reference_mol = src_mols_by_name[REFERENCE_MOL_NAME]
    csv_rows.append(
        (
            REFERENCE_MOL_NAME,
            Chem.MolToSmiles(reference_mol),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(reference_mol.GetProp(GLIDE_SCORE_PROPERTY)),
            0,
        )
    )
    for result_path in results_dir.glob("*.json"):
        with open(result_path, "r") as ifs:
            results = json.load(ifs)
        for data in results["edges"]:
            if "complex_pred_ddg" not in data or "solvent_pred_ddg" not in data:
                continue
            assert (
                data["mol_a"] == REFERENCE_MOL_NAME
            ), f"Unexpected mol a: {data['mol_a']}"
            mol_name = data["mol_b"]
            mol = src_mols_by_name[mol_name]
            pred_ddg = data["complex_pred_ddg"] - data["solvent_pred_ddg"]
            pred_ddg_err = np.linalg.norm(
                [data["complex_pred_ddg_err"], data["complex_pred_ddg_err"]]
            )
            num_dummy_atoms = reference_mol.GetNumAtoms() + mol.GetNumAtoms() - (2 * len(data["core"]))
            csv_rows.append(
                (
                    data["mol_b"],
                    Chem.MolToSmiles(mol),
                    pred_ddg,
                    pred_ddg_err,
                    data["vacuum_pred_ddg"],
                    data["solvent_pred_ddg"],
                    data["complex_pred_ddg"],
                    float(mol.GetProp(GLIDE_SCORE_PROPERTY)),
                    num_dummy_atoms,
                )
            )

    headers = ["mol_name", "SMILES", "dG_bind", "dG_bind_err", "vacuum_dG_bind", "solvent_dG_bind", "complex_dG_bind", "Glide_score", "dummy_atoms"]

    with open(results_dir / "active_learning_inputs.csv", "w", newline="") as ofs:
        csv_writer = csv.writer(ofs)
        csv_writer.writerow(headers)
        for row in sorted(csv_rows, key=lambda x: x[2]):
            csv_writer.writerow(row)


if __name__ == "__main__":
    main()
