import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import trange
def segment_grid_and_color(mask_array, grid_size, pads=[0]):
    """
    Segments a 2D binary mask array into grids and colors the grids containing label 1 with random colors.

    Parameters:
    - mask_array: 2D binary mask array containing 0s and 1s.
    - grid_size: Size of the grid.
    - pads: A list specifying the padding for each grid.

    Returns:
    - colored_mask: 3D RGB color mask.
    """
    # Focus on regions with label 1
    # mask_array = (mask_array == 1).astype(int)
    # Ensure the input mask is binary and we focus on regions marked as 1
    rows, cols = mask_array.shape
    mask_array = mask_array.astype(int).reshape(rows, cols, 1)
    colored_mask = np.zeros((rows, cols, 3), dtype=np.uint8)  # Create RGB color mask
    # Generate random colors for grid size
    for i in range(0, rows, grid_size):
        for j in range(0, cols, grid_size):
            # Check if this grid contains any pixels with label 1
            if np.any(mask_array[i:i+grid_size, j:j+grid_size] == 1):
                for pad in pads:
                    random_color = np.concatenate([np.random.randint(0, 256, 3)], axis=0)
                    # Apply random color to the grid, excluding the padding part
                    colored_mask[i + pad:i + grid_size - pad, j + pad:j + grid_size - pad] = random_color * mask_array[
                                                                                                            i + pad:i + grid_size - pad,
                                                                                                            j + pad:j + grid_size - pad]
    return colored_mask


def dilate_mask_and_color(mask_array, iterations=10, structure=None):
    """
    Dilates a binary mask using a specified structuring element and colors the dilated regions with a random color.

    Parameters:
    - mask_array: 2D binary mask array containing 0s and 1s.
    - iterations: Number of dilation iterations.
    - structure: Structuring element for dilation. Defaults to a 3x3 rectangular kernel if None.

    Returns:
    - colored_mask: 3D RGB color mask.
    """
    # Ensure the input mask is binary and we focus on regions marked as 1
    rows, cols = mask_array.shape
    # The following line is commented out because it's a step that might not be necessary depending on the input format
    # mask_array = mask_array.astype(int).reshape(rows, cols, 1)

    # Define different types of structuring elements for dilation
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Rectangular kernel
    # ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical kernel
    # cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # Cross-shaped kernel

    # Set the default structuring element to rectangular if none is specified
    if structure is None:
        structure = rect_kernel

    # Perform the dilation operation on the binary mask
    dilated_mask = binary_dilation(mask_array, structure=structure, iterations=iterations)

    # Generate a random RGB color for the dilated regions
    random_color = np.concatenate([np.random.randint(0, 256, 3)], axis=0)

    # Apply the random color to the dilated regions to create the final colored mask
    colored_mask = random_color * dilated_mask.reshape(rows, cols, 1)

    # Return the colored mask
    return colored_mask


def segment_and_dilate(mask_generator, img, grid_rate, dilate_iter):
    """
    Processes an image to segment and dilate masks, then colors them. It reads an image, generates masks,
    and applies grid-based coloring and dilation based on the specified parameters.

    Parameters:
    - mask_generator: Object responsible for generating masks from the image.
    - img_path: Path to the input image.
    - grid_rate: Rate used to determine grid size.
    - dilate_iter: Number of dilation iterations.

    Returns:
    - colored_mask: 3D RGB color mask.
    """

    # Generate masks for the image
    masks = mask_generator.generate(img)

    # Initialize the color mask with zeros
    img_width, img_height = img.shape[1], img.shape[0]
    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Calculate grid size based on the shorter side of the image and grid_rate
    grid_size = np.floor(np.min([img_width, img_height]) * grid_rate).astype(np.int32)

    # Process each generated mask
    mask_a = []
    for j in trange(len(masks)):
        # If the area of the mask is larger than the grid size, add it to mask_a for further processing
        if masks[j]['segmentation'].sum() > (grid_size ** 2):
            mask_a.append(masks[j]['segmentation'])
        else:
            # Dilate and color smaller masks, then add them to the color mask
            colored_mask = colored_mask + dilate_mask_and_color(masks[j]['segmentation'], iterations=dilate_iter) * (colored_mask == 0)

    # If mask_a is not empty, merge and color the masks in mask_a using a grid method
    if len(mask_a) > 0:
        mask_a = np.array(mask_a).sum(0).astype(np.bool_)
        colored_mask = colored_mask + segment_grid_and_color(mask_a, grid_size=grid_size, pads=[0]) * (colored_mask == 0)

    # Limit the pixel values of the color mask to the 0-255 range
    colored_mask = colored_mask.clip(0, 255).astype(np.uint8)

    # Return the final color mask
    return colored_mask




