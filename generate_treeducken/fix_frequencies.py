import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# Folder containing .tgl files
FOLDER = "../generate_treeducken/generated_trees/Datasets/"

# Regex to extract key-value lines like:
pattern = re.compile(
    r"^(Cospeciations|Host_Speciations|Host_Extinctions|Symbiont_Speciations|Symbiont_Extinctions|Host_Spread/Switches|Total Events)\s+([0-9\.NaNnan]+)",
    re.MULTILINE
)

def process_file(path):
    try:
        with open(path, "r") as f:
            content = f.read()

        # Extract all needed values
        matches = dict(re.findall(pattern, content))

        def safe_float(x):
            try:
                return float(x)
            except:
                return None

        # Extract values
        cos = safe_float(matches.get("Cospeciations"))
        hs  = safe_float(matches.get("Host_Speciations"))
        switch = safe_float(matches.get("Host_Spread/Switches"))
        he  = safe_float(matches.get("Host_Extinctions"))
        ss  = safe_float(matches.get("Symbiont_Speciations"))
        se  = safe_float(matches.get("Symbiont_Extinctions"))
        total = safe_float(matches.get("Total Events"))

        # Replace NaN/None with zero
        def fix_nan(x):
            return 0 if (x is None or (isinstance(x, float) and x != x)) else x

        cos, hs, switch, he, ss, se, total = map(fix_nan, [cos, hs, switch, he, ss, se, total])

        # Convert event types using OLD total
        hs = round(hs * total)
        he = round(he * total)
        ss = round(ss * total)
        se = round(se * total)
        cos = round(cos * total)
        switch = round(switch * total)

        # New total (only these two events matter)
        new_total = cos + switch

        new_cos_freq = cos / new_total if new_total > 0 else 0
        new_switch_freq = switch / new_total if new_total > 0 else 0

        new_values = {
            "Cospeciations":         new_cos_freq,
            "Host_Spread/Switches":  new_switch_freq,
            "Host_Speciations":      hs,
            "Host_Extinctions":      he,
            "Symbiont_Speciations":  ss,
            "Symbiont_Extinctions":  se,
            "Total Events":          new_total
        }

        # Replace values in file
        for key, new_val in new_values.items():
            content = re.sub(
                rf"({key}\s+)[0-9\.NaNnan]+",
                rf"\g<1>{new_val:.6f}",
                content
            )

        with open(path, "w") as f:
            f.write(content)

        return None  # success

    except Exception as e:
        return f"ERROR in {path}: {e}"


def main():
    files = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.endswith(".tgl")]

    print(f"Found {len(files)} .tgl files")

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_workers} workers")

    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            if result:  # if error
                errors.append(result)

    print("\nDone!")

    if errors:
        print("\n⚠️ ERRORS FOUND:")
        for err in errors:
            print(err)


if __name__ == "__main__":
    main()