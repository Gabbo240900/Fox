#!/usr/bin/env python3
import os
import glob
import argparse
import torch


def write_tgl(out_path, host_msas, parasite_msas, mappings, event_frequencies):
    """
    Writes a .tgl compatible with your CophylogenyDataset._parse_tgl_file().
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Ensure deterministic ordering (nice for diffs)
    host_items = sorted(host_msas.items(), key=lambda x: x[0])
    para_items = sorted(parasite_msas.items(), key=lambda x: x[0])

    # mappings expected as list of (parasite, host)
    # ensure strings, and keep only P\d / H\d style if you want strictness
    cleaned_mappings = []
    for p, h in mappings:
        p = str(p).strip().strip(",")
        h = str(h).strip().strip(",")
        cleaned_mappings.append((p, h))

    with open(out_path, "w") as f:
        # ---- HOST ----
        f.write("BEGIN HOST;\n")
        # Optional: frequencies can be anywhere; your parser scans all lines
        if "Cospeciations" in event_frequencies:
            f.write(f"Cospeciations {event_frequencies['Cospeciations']}\n")
        if "Host_spread/Switches" in event_frequencies:
            f.write(f"Host_Spread/Switches {event_frequencies['Host_spread/Switches']}\n")
        if "Sim_time" in event_frequencies:
            f.write(f"Sim_time {event_frequencies['Sim_time']}\n")

        for sp, seq in host_items:
            f.write(f"{sp} {seq}\n")
        f.write("ENDBLOCK;\n\n")

        # ---- PARASITE ----
        f.write("BEGIN PARASITE;\n")
        for sp, seq in para_items:
            f.write(f"{sp} {seq}\n")
        f.write("ENDBLOCK;\n\n")

        # ---- DISTRIBUTION / MAPPING ----
        f.write("BEGIN DISTRIBUTION;\n")
        # Your parser expects: "P123: H45"
        for p, h in cleaned_mappings:
            f.write(f"{p}: {h}\n")
        f.write("ENDBLOCK;\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing .pt files")
    ap.add_argument("--out_dir", required=True, help="Directory to write .tgl files")
    ap.add_argument("--pattern", default="*.pt", help="Glob pattern (default: *.pt)")
    args = ap.parse_args()

    pt_files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not pt_files:
        raise SystemExit(f"No .pt files found in {args.in_dir} with pattern {args.pattern}")

    ok = 0
    skip = 0

    for pt_path in pt_files:
        try:
            s = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[SKIP] {pt_path} (load failed: {e})")
            skip += 1
            continue

        # Required by your pipeline
        host_msas = s.get("host_msas", None)
        parasite_msas = s.get("parasite_msas", None)
        mappings = s.get("mappings", None)
        event_frequencies = s.get("event_frequencies", {}) or {}

        if not isinstance(host_msas, dict) or not isinstance(parasite_msas, dict) or mappings is None:
            print(f"[SKIP] {os.path.basename(pt_path)} missing host_msas/parasite_msas/mappings")
            skip += 1
            continue

        base = os.path.splitext(os.path.basename(pt_path))[0]
        out_path = os.path.join(args.out_dir, base + ".tgl")

        write_tgl(out_path, host_msas, parasite_msas, mappings, event_frequencies)
        ok += 1

    print(f"\nDone. Wrote {ok} .tgl files. Skipped {skip}.")


if __name__ == "__main__":
    main()