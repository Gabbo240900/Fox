import os
import re
import json
import numpy as np
from torch.utils.data import Dataset

class CophylogenyDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_paths = [
                entry.path
                for entry in os.scandir(dataset_dir)
                if entry.is_file() and entry.name.lower().endswith(".tgl")
            ]
    def _parse_tgl_file(self, dataset_path):
        with open(dataset_path, "r") as file:
            lines = file.readlines()

        section = None
        current_dataset = {
            "host_msas": {},
            "parasite_msas": {},
            "mappings": [],
            "event_frequencies": {}
        }

        event_name_map = {
            "Cospeciations": "Cospeciations",
            "Host_Spread/Switches": "Host_spread/Switches",
            "Sim_time": "Sim_time",
        }

        msa_dict = None
        for raw in lines:
            line = raw.strip()

            if line.startswith("BEGIN HOST;"):
                section = "HOST"
                msa_dict = current_dataset["host_msas"]
                continue
            elif line.startswith("BEGIN PARASITE;"):
                section = "PARASITE"
                msa_dict = current_dataset["parasite_msas"]
                continue
            elif line.startswith("BEGIN DISTRIBUTION;"):
                section = "MAPPING"
                msa_dict = None
                continue
            elif line.startswith("ENDBLOCK;"):
                section = None
                msa_dict = None
                continue

            # Event frequencies
            if any(ev in line for ev in event_name_map):
                match = re.match(r"([A-Za-z_/]+)\s+([-\d.eE+,]+)", line)
                if match:
                    raw_key, value = match.groups()
                    std_key = event_name_map.get(raw_key.strip())
                    if std_key:
                        try:
                            value = float(value.replace(",", "."))
                            current_dataset["event_frequencies"][std_key] = value
                        except ValueError:
                            pass
                continue

            # MSAs
            if section in ("HOST", "PARASITE"):
                if line.endswith("'"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    species, sequence = parts
                    msa_dict[species] = sequence
                continue

            # Mappings
            if section == "MAPPING" and re.match(r"P\d+: H\d+", line):
                parasite, host = line.split(": ")
                current_dataset["mappings"].append((parasite.strip(","), host.strip(",")))
                continue

        return current_dataset

    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0 or index >= len(self.dataset_paths):
                raise IndexError("Dataset index out of range")
            return self._parse_tgl_file(self.dataset_paths[index])
        raise TypeError("Index must be an integer")

    def get_data(self, index=None):
        if index is None:
            return self
        return self.__getitem__(index)
