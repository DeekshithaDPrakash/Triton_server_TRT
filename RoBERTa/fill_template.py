import os
import argparse


def generate_model_config(output_path, model_name, max_batch_size, backend, platform, input_config, output_config, parameters=None):
    """
    Generate a Triton config.pbtxt for a model.
    """
    config = f"""
name: "{model_name}"
"""
    if backend == "python":
        config += f'backend: "python"\n'
    else:
        config += f'platform: "{platform}"\n'

    config += f"max_batch_size: {max_batch_size}\n\n"

    # Add input configurations
    config += "input [\n"
    for inp in input_config:
        config += f"""  {{
    name: "{inp['name']}"
    data_type: {inp['data_type']}
    dims: [{", ".join(inp['dims'])}]
  }},\n"""
    config = config.rstrip(",\n") + "\n]\n"

    # Add output configurations
    config += "output [\n"
    for out in output_config:
        config += f"""  {{
    name: "{out['name']}"
    data_type: {out['data_type']}
    dims: [{", ".join(out['dims'])}]
  }},\n"""
    config = config.rstrip(",\n") + "\n]\n"

    # Add parameters (if any)
    if parameters:
        config += "parameters {\n"
        for key, value in parameters.items():
            config += f"""  key: "{key}"
  value: {{
    string_value: "{value}"
  }}\n"""
        config += "}\n"

    # Save the config to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(config)
    print(f"Config for '{model_name}' generated at: {output_path}")

    # Create version folder for the model
    version_dir = os.path.join(os.path.dirname(output_path), "1")
    os.makedirs(version_dir, exist_ok=True)
    print(f"Version directory created for '{model_name}' at: {version_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Triton config.pbtxt")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save config.pbtxt")
    parser.add_argument("-n", "--model_name", type=str, required=True, help="Model name for Triton")
    parser.add_argument("-b", "--max_batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "tensorrt_plan"], help="Backend type")
    parser.add_argument("--platform", type=str, default="", help="Platform (only for non-Python models)")
    parser.add_argument("-i", "--input_config", type=str, required=True, help="Input configuration (name:data_type:dims)")
    parser.add_argument("-u", "--output_config", type=str, required=True, help="Output configuration (name:data_type:dims)")
    parser.add_argument("-p", "--parameters", type=str, help="Additional parameters (key:value)")

    args = parser.parse_args()

    # Parse input configurations
    input_config = [
        {"name": item.split(":")[0], "data_type": item.split(":")[1], "dims": item.split(":")[2].split(",")}
        for item in args.input_config.split(";")
    ]
    # Parse output configurations
    output_config = [
        {"name": item.split(":")[0], "data_type": item.split(":")[1], "dims": item.split(":")[2].split(",")}
        for item in args.output_config.split(";")
    ]
    # Parse parameters
    parameters = dict(param.split(":") for param in args.parameters.split(",")) if args.parameters else None

    # Generate the config.pbtxt file
    generate_model_config(
        args.output_path,
        args.model_name,
        args.max_batch_size,
        args.backend,
        args.platform,
        input_config,
        output_config,
        parameters,
    )
