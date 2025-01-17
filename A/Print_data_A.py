import numpy as np
import matplotlib.pyplot as plt

def print_npz_contents(npz_path):
    """
    Prints the contents of an .npz file, including key names, shapes, data types,
    and samples of the data.

    Parameters:
    - npz_path (str): The file path to the .npz file.
    """
    try:
        # Load the .npz file
        with np.load(npz_path) as data:
            # Iterate through each key in the .npz file
            for key in data.files:
                array = data[key]
                print(f"Key: '{key}'")
                print(f" - Shape: {array.shape}")
                print(f" - Data Type: {array.dtype}")
                
                # Optionally, print a summary or a sample of the data
                if 'images' in key:
                    # For image data, print the first image's shape and type
                    first_image = array[0]
                    print(f" - First image shape: {first_image.shape}")
                    print(f" - First image data type: {first_image.dtype}")
                    print(f" - First image data (first pixel): {first_image.flatten()[0]}")
                    
                    # Optionally, visualize the first image
                    visualize = input("Do you want to visualize the first image? (y/n): ").strip().lower()
                    if visualize == 'y':
                        if first_image.ndim == 2:
                            plt.imshow(first_image, cmap='gray')
                        elif first_image.ndim == 3:
                            if first_image.shape[-1] == 3:
                                plt.imshow(first_image)
                            else:
                                # For 3D data, visualize the first slice
                                plt.imshow(first_image[:, :, 0], cmap='gray')
                        plt.title(f"First Image in {key}")
                        plt.axis('off')
                        plt.show()
                elif 'labels' in key:
                    # For label data, print the first few labels
                    first_labels = array[:5].flatten()
                    print(f" - First 5 labels: {first_labels}")
                print("-" * 50)
    except FileNotFoundError:
        print(f"Error: The file '{npz_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    data_dir = r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\BreastMNIST.npz'  # Replace with your .npz file path
    print_npz_contents(data_dir)