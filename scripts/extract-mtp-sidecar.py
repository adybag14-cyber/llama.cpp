#!/usr/bin/env python3
"""Extract a name-preserving MTP sidecar from a combined GGUF model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gguf


def field_value(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise KeyError(f"required GGUF metadata field is missing: {key}")
    return field.contents()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract token/output tensors and one MTP block without renumbering its layer names",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--layer",
        type=int,
        help="MTP layer index (default: general block count minus one)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve(strict=True)
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(input_path, "r")
    arch = field_value(reader, gguf.Keys.General.ARCHITECTURE)
    block_count = int(field_value(reader, f"{arch}.block_count"))
    layer = block_count - 1 if args.layer is None else args.layer
    if layer < 0 or layer >= block_count:
        raise ValueError(f"MTP layer {layer} is outside [0, {block_count})")

    globals_to_keep = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    }
    layer_prefix = f"blk.{layer}."
    tensors = [
        tensor
        for tensor in reader.tensors
        if tensor.name in globals_to_keep or tensor.name.startswith(layer_prefix)
    ]
    names = {tensor.name for tensor in tensors}
    missing = globals_to_keep - names
    if missing:
        raise KeyError(f"required global tensors are missing: {sorted(missing)}")
    if not any(name.startswith(f"{layer_prefix}nextn.") for name in names):
        raise KeyError(f"layer {layer} does not contain NextN/MTP tensors")

    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(f"refusing to overwrite existing partial output: {partial_path}")

    writer = gguf.GGUFWriter(partial_path, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = int(alignment.contents())

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    for tensor in tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        writer.close()
        os.replace(partial_path, output_path)
    except BaseException:
        try:
            writer.close()
        finally:
            raise

    total_bytes = sum(tensor.n_bytes for tensor in tensors)
    print(f"wrote {len(tensors)} tensors ({total_bytes} tensor bytes) from layer {layer} to {output_path}")


if __name__ == "__main__":
    main()
