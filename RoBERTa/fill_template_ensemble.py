import os
import argparse

def generate_ensemble_config(output_path, model_name, max_batch_size, input_config, output_config, ensemble_steps):
    """
    Generate a Triton config.pbtxt for an ensemble model.
    """
    config = f"""
name: "{model_name}"
platform: "ensemble"
max_batch_size: {max_batch_size}

input [
"""
    for inp in input_config:
        config += f"""  {{
    name: "{inp['name']}"
    data_type: {inp['data_type']}
    dims: [{", ".join(inp['dims'])}]
  }},
"""
    config = config.rstrip(",\n") + "\n]\n"

    config += "output [\n"
    for out in output_config:
        config += f"""  {{
    name: "{out['name']}"
    data_type: {out['data_type']}
    dims: [{", ".join(out['dims'])}]
  }},
"""
    config = config.rstrip(",\n") + "\n]\n"

    config += "ensemble_scheduling {\n  step [\n"
    for step in ensemble_steps:
        config += f"""    {{
      model_name: "{step['model_name']}"
      model_version: {step['model_version']}
"""
        # Add input maps
        for key, value in step["input_map"].items():
            config += f"""      input_map {{
        key: "{key}"
        value: "{value}"
      }}
"""
        # Add output maps
        for key, value in step["output_map"].items():
            config += f"""      output_map {{
        key: "{key}"
        value: "{value}"
      }}
"""
        config += "    },\n"
    config = config.rstrip(",\n") + "\n  ]\n}\n"

    # Ensure output directory and save config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(config.strip())
    print(f"Ensemble config for '{model_name}' generated at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Triton ensemble config.pbtxt")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save config.pbtxt")
    parser.add_argument("-n", "--model_name", type=str, required=True, help="Ensemble model name for Triton")
    parser.add_argument("-b", "--max_batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("-i", "--input_config", type=str, required=True, help="Input configuration (name:data_type:dims)")
    parser.add_argument("-u", "--output_config", type=str, required=True, help="Output configuration (name:data_type:dims)")
    parser.add_argument("-e", "--ensemble_steps", type=str, required=True, help="Ensemble steps in the format: model_name:model_version:input_maps:output_maps;...")

    args = parser.parse_args()

    # Parse input configurations
    input_config = []
    for item in args.input_config.split(";"):
        parts = item.split(":")
        input_config.append({
            "name": parts[0],
            "data_type": parts[1],
            "dims": parts[2].split(","),
        })

    # Parse output configurations
    output_config = []
    for item in args.output_config.split(";"):
        parts = item.split(":")
        output_config.append({
            "name": parts[0],
            "data_type": parts[1],
            "dims": parts[2].split(","),
        })

    # Parse ensemble steps
    ensemble_steps = []
    for step in args.ensemble_steps.strip(';').split(";"):
        parts = step.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid ensemble step format: {step}")
        model_name = parts[0]
        model_version = int(parts[1])

        # Parse input_map
        input_map_entries = parts[2].split(",") if parts[2] else []
        input_map = dict(entry.split("=") for entry in input_map_entries)

        # Parse output_map
        output_map_entries = parts[3].split(",") if parts[3] else []
        output_map = dict(entry.split("=") for entry in output_map_entries)

        ensemble_steps.append({
            "model_name": model_name,
            "model_version": model_version,
            "input_map": input_map,
            "output_map": output_map,
        })

    # Generate the ensemble config
    generate_ensemble_config(
        args.output_path,
        args.model_name,
        args.max_batch_size,
        input_config,
        output_config,
        ensemble_steps,
    )
