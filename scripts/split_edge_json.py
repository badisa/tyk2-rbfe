import json
from argparse import ArgumentParser
from pathlib import Path

from timemachine.utils import batches


def main():
    parser = ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("num_parts", type=int)
    args = parser.parse_args()

    assert args.num_parts > 1, "Need to generate at least two parts"

    path = Path(args.input_path).expanduser()
    assert path.is_file()

    with open(path, "r") as ifs:
        edges = json.load(ifs)

    batch_size = len(edges) // args.num_parts

    offset = 0
    for i, batch in enumerate(batches(len(edges), batch_size)):
        base, ext = str(path).split(".")
        output_path = f"{base}_part_{i}.json"
        print(f"Writing out part {i} to {output_path}")
        with open(output_path, "w") as ofs:
            json.dump(edges[offset : offset + batch], ofs, indent=2)
        print(batch)
        offset += batch


if __name__ == "__main__":
    main()
