import os


def rescale_obj_mesh(input_path, scale_factor, output_path=None):
    """
    Rescale an OBJ mesh by a given scale factor.

    Args:
        input_path (str): Path to input OBJ file
        scale_factor (float): Scale factor to apply to the mesh
        output_path (str, optional): Path to save the rescaled mesh. If None,
                                   will append '_scaled' to the input filename
    """
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scaled{ext}"

    # Read the input file and process line by line
    with open(input_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue

            # Split line into components
            parts = line.split()
            if not parts:
                continue

            # Scale vertex coordinates (lines starting with 'v')
            if parts[0] == "v":
                try:
                    # Get the vertex coordinates
                    x, y, z = map(float, parts[1:4])
                    # Scale the coordinates
                    scaled_vertex = f"v {x * scale_factor} {y * scale_factor} {z * scale_factor}"
                    # Add any additional vertex data (like colors) if present
                    if len(parts) > 4:
                        scaled_vertex += " " + " ".join(parts[4:])
                    outfile.write(scaled_vertex + "\n")
                except (ValueError, IndexError):
                    # If there's any error parsing the vertex line, write it unchanged
                    outfile.write(line + "\n")
            else:
                # Write all non-vertex lines unchanged
                outfile.write(line + "\n")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rescale an OBJ mesh file")
    parser.add_argument("--input_path", help="Path to input OBJ file")
    parser.add_argument("--scale_factor", type=float, help="Scale factor to apply to the mesh")
    parser.add_argument("--output_path", "-o", help="Path to save the rescaled mesh (optional)")

    args = parser.parse_args()

    output_file = rescale_obj_mesh(args.input_path, args.scale_factor, args.output_path)
    print(f"Rescaled mesh saved to: {output_file}")
