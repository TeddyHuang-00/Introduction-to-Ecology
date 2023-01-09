import os
import json
import subprocess

python_ver = "python3.11"
default_config = {
    "name": "default",
    "prey": [500, 500],
    "toxicity": [0.9, 0.0],
    "predator": 100,
    "genes": 10,
    "generations": 1000,
    "mutation_rate": 0.05,
    "mutation_scale": 0.05,
    "initial_scale": 0.1,
    "seed": 0,
    "cmap": "rainbow",
}

configs = [default_config | c for c in json.loads(open("./batch-config.json").read())]

for config in configs:
    print(config)
    if (
        os.path.exists(os.path.join("fig", config["name"], "Population.mp4"))
        and os.path.exists(os.path.join("fig", config["name"], "Distance.mp4"))
        and not config.get("force", False)
    ):
        print("Output directory exists, skipping")
        continue
    subprocess.call(
        [
            python_ver,
            "simulate.py",
            "-P",
            *map(str, config["prey"]),
            "-T",
            *map(str, config["toxicity"]),
            "-H",
            str(config["predator"]),
            "-G",
            str(config["genes"]),
            "-N",
            str(config["generations"]),
            "-R",
            str(config["mutation_rate"]),
            "-K",
            str(config["mutation_scale"]),
            "-I",
            str(config["initial_scale"]),
            "-S",
            str(config["seed"]),
            "-C",
            config["cmap"],
            "-O",
            os.path.join("fig", config["name"]),
        ]
    )
    subprocess.call(
        [
            python_ver,
            "post-process.py",
            "-F",
            "-D",
            os.path.join("fig", config["name"]),
        ]
    )
