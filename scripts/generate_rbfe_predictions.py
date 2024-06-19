import json
import time
from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path

# Noticed that after running in a loop for a long time, memory runs out due to open plots
# See https://github.com/matplotlib/matplotlib/issues/20300 for the horrible reality of using
# GUI backends
import matplotlib
import numpy as np
from rdkit import Chem
from timemachine import constants
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import WaterSamplingParams
from timemachine.fe.rbfe import (
    DEFAULT_HREX_PARAMS,
    MDParams,
    run_complex,
    run_solvent,
    run_vacuum,
)
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.parallel.client import CUDAPoolClient
from timemachine.parallel.utils import get_gpu_count
from timemachine.potentials.jax_utils import pairwise_distances

matplotlib.use("agg")


def main():
    parser = ArgumentParser(description="Generate star map as a JSON file")
    parser.add_argument("sdf_file", help="Path to sdf file containing mols")
    parser.add_argument("pdb_file", help="Path to pdb file containing structure")
    parser.add_argument("edges_json", help="Json list containing pairs of mol names")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument(
        "output_json", help="Output json file, if exists will concatenate to it"
    )
    parser.add_argument("--forcefield", default="smirnoff_2_2_0_ccc.py")
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
    min_overlap = 0.333
    n_windows = 24
    if args.testing:
        md_params = replace(
            md_params,
            n_eq_steps=100,
            n_frames=100,
            water_sampling_params=WaterSamplingParams(),
        )

        # Fixed number of windows
        n_windows = 2
    else:
        md_params = replace(
            md_params,
            n_eq_steps=200_000,
            n_frames=2000,
            water_sampling_params=WaterSamplingParams(),
        )

    def write_output_data():
        with open(output_path, "w") as ofs:
            json.dump(data, ofs, indent=2)

    data = {"edges": []}
    output_path = Path(args.output_json).expanduser()
    if output_path.is_file():
        with open(output_path, "r") as ifs:
            data = json.load(ifs)
    if "md_params" in data:
        assert data["md_params"]["n_eq_steps"] == md_params.n_eq_steps
        assert data["md_params"]["n_frames"] == md_params.n_frames
        assert data["md_params"]["steps_per_frame"] == md_params.steps_per_frame
        assert data["md_params"]["seed"] == md_params.seed
        assert data["md_params"]["local_steps"] == md_params.local_steps
    else:
        data["md_params"] = asdict(md_params)
    if "max_windows" in data:
        assert data["max_windows"] == n_windows
    else:
        data["max_windows"] = n_windows

    if "min_overlap" in data:
        assert data["min_overlap"] == min_overlap
    else:
        data["min_overlap"] = min_overlap
    write_output_data()

    def update_output_data_with_edge_data(edge_data):
        # Janky handling of the core, as SingleToplogy expects np.array
        # and JSON expects list of lists
        edge_data["core"] = [list([int(y) for y in x]) for x in edge_data["core"]]
        index_to_update = -1
        edges = data["edges"]
        for i in range(len(edges)):
            if (
                edges[i]["mol_a"] == edge_data["mol_a"]
                and edges[i]["mol_b"] == edge_data["mol_b"]
            ):
                index_to_update = i
                break
        if index_to_update >= 0:
            edges[index_to_update] = edge_data
            data["edges"] = edges
        else:
            edges.append(edge_data)
        write_output_data()

    # JSON Edge format
    # {
    #   "mol_a": "name_a",
    #   "mol_b": "name_b",
    #   "core": list of pairs
    #   "<leg_name>__pred_ddg": float
    #   "<leg_name>_pred_ddg_err": float
    #   "<leg_name>_windows": int
    # }

    existing_edges = {(x["mol_a"], x["mol_b"]): x for x in data.get("edges", [])}

    for mol_name_a, mol_name_b in edges:
        mol_a = mols_by_name[mol_name_a]
        # Using an old copy of the charge cache, just pretend like the are ELF10
        mol_a.SetProp("AM1ELF10Cache", mol_a.GetProp("AM1Cache"))
        mol_b = mols_by_name[mol_name_b]
        mol_b.SetProp("AM1ELF10Cache", mol_b.GetProp("AM1Cache"))

        if (mol_name_a, mol_name_b) in existing_edges:
            edge_data = existing_edges[(mol_name_a, mol_name_b)]
        else:
            edge_data = {
                "mol_a": mol_name_a,
                "mol_b": mol_name_b,
            }

        if "core" not in edge_data:
            edge_data["core"] = atom_mapping.get_cores(
                mol_a, mol_b, **constants.DEFAULT_ATOM_MAPPING_KWARGS
            )[0]
            update_output_data_with_edge_data(edge_data)
        core = np.array(edge_data["core"])
        for leg_name in ["vacuum", "solvent", "complex"]:
            if f"{leg_name}_pred_ddg" not in edge_data:
                if leg_name == "vacuum":
                    res = run_vacuum(
                        mol_a,
                        mol_b,
                        core,
                        ff,
                        None,
                        md_params,
                        n_windows=n_windows,
                        min_overlap=min_overlap,
                    )
                elif leg_name == "solvent":
                    res, _, _ = run_solvent(
                        mol_a,
                        mol_b,
                        core,
                        ff,
                        None,
                        md_params,
                        n_windows=n_windows,
                        min_overlap=min_overlap,
                    )
                elif leg_name == "complex":
                    res, _, _ = run_complex(
                        mol_a,
                        mol_b,
                        core,
                        ff,
                        str(Path(args.pdb_file).expanduser()),
                        md_params,
                        n_windows=n_windows,
                        min_overlap=min_overlap,
                    )
                edge_data[f"{leg_name}_pred_ddg"] = float(np.sum(res.final_result.dGs))
                edge_data[f"{leg_name}_pred_ddg_err"] = float(
                    np.linalg.norm(res.final_result.dG_errs)
                )
                edge_data[f"{leg_name}_windows"] = len(res.final_result.initial_states)
                edge_data[f"{leg_name}_min_overlap"] = min(
                    bar.overlap for bar in res.final_result.bar_results
                )
                update_output_data_with_edge_data(edge_data)


if __name__ == "__main__":
    main()
