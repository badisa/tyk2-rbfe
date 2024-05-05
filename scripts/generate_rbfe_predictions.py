import json
import time
import numpy as np
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

from rdkit import Chem
from timemachine import constants
from timemachine.md import builders
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import WaterSamplingParams
from timemachine.fe.rbfe import (DEFAULT_HREX_PARAMS, run_complex, run_solvent,
                                 run_vacuum)
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield


def main():
    parser = ArgumentParser(description="Generate star map as a JSON file")
    parser.add_argument("sdf_file", help="Path to sdf file containing mols")
    parser.add_argument("edges_json", help="Json list containing pairs of mol names")
    parser.add_argument("--forcefield", default=constants.DEFAULT_FF)
    args = parser.parse_args()

    with open(args.edges_json, "r") as ifs:
        edges = json.load(ifs)
    assert len(edges) > 0, "Got empty edges"
    assert all(len(e) == 2 for e in edges), "All edges must be pairs"

    mols = read_sdf(args.sdf_file)
    mols_by_name = {mol.GetProp("_Name"): mol for mol in mols}

    ff = Forcefield.load_from_file(args.forcefield)

    # Fixed MD Params
    md_params = DEFAULT_HREX_PARAMS
    md_params = replace(
        md_params,
        n_eq_steps=200_000,
        n_frames=2000,
        local_steps=md_params.steps_per_frame,
        water_sampling_params=WaterSamplingParams(),
    )

    # Fixed min overlap
    min_overlap = 0.333
    # Fixed number of windows
    n_windows = 30

    for mol_name_a, mol_name_b in edges:
        mol_a = mols_by_name[mol_name_a]
        mol_b = mols_by_name[mol_name_b]

        core = atom_mapping.get_cores(mol_a, mol_b, **constants.DEFAULT_ATOM_MAPPING_KWARGS)[0]
        start = time.time()
        res = run_vacuum(mol_a, mol_b, core, ff, None, md_params, n_windows=n_windows, min_overlap=min_overlap)
        end = time.time()
        print("Took", end - start)
        ddg_pred = np.sum(res.final_result.dGs)
        ddg_err = np.linalg.norm(res.final_result.dG_errs)
        print("Pred", ddg_pred, "Err", ddg_err, "Took", end - start)
        start = time.time()
        res, _, _ = run_solvent(mol_a, mol_b, core, ff, None, md_params, n_windows=n_windows, min_overlap=min_overlap)
        end = time.time()
        print("Took", end - start)
        ddg_pred = np.sum(res.final_result.dGs)
        ddg_err = np.linalg.norm(res.final_result.dG_errs)
        print("Pred", ddg_pred, "Err", ddg_err, "Took", end - start)
        break


if __name__ == '__main__':
    main()