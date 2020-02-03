"""Create a new validation.json file, loading in from defaults and making some adjustments for runtime and consistency
"""
from pathlib import Path
import json

# Load the default config
defaults_path = Path.cwd().parent / "hawks" / "defaults.json"
with open(defaults_path) as json_file:
    config = json.load(json_file)
# Make modifications for reproducible behaviour (and quicker tests)
config["hawks"]["seed_num"] = 1
config["hawks"]["num_runs"] = 2
config["dataset"]["num_examples"] = 500
config["ga"]["num_gens"] = 20
print(config["hawks"])
# Save the config
fpath = "validation.json"
with open(fpath, "w") as f:
    json.dump(config, f, indent=4)
