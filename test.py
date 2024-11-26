import ast
import json


# Function to read from a .txt file and convert it into an array of arrays
def convert_txt_to_json(txt_filename="obstacles.txt", json_filename="obstacles.json"):
    # Initialize an empty list to hold the obstacles as arrays
    obstacles = []

    try:
        with open(txt_filename, "r") as f:
            for line in f:
                # Remove whitespace and newline characters
                line = line.strip()

                # Check if the line is not empty
                if line:
                    try:
                        # Use ast.literal_eval to safely evaluate the list of tuples
                        obstacle_list = ast.literal_eval(line)

                        # Check if the evaluation resulted in a list of tuples
                        if isinstance(obstacle_list, list) and all(
                                isinstance(item, tuple) and len(item) == 2 for item in obstacle_list):
                            obstacles.extend(obstacle_list)  # Add all tuples from this line to the obstacles list
                        else:
                            print(f"Skipping invalid line (not a list of tuples): {line}")

                    except (ValueError, SyntaxError) as e:
                        # Print the error if there is an issue with the line format
                        print(f"Error processing line: {line}, error: {e}")

        # Save the list of obstacles as a JSON file
        with open(json_filename, "w") as json_file:
            json.dump(obstacles, json_file)

        print(f"Successfully converted {txt_filename} to {json_filename}")

    except FileNotFoundError:
        print(f"Error: The file {txt_filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function to convert the .txt file to .json
convert_txt_to_json("static_obstacles.txt", "obstacles.json")
