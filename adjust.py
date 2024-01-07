def adjust_bbox(bbox, old_shape):
    # Unpack the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Get the width and height of the old shape
    old_height, old_width = old_shape[:2]

    # Calculate the scaling factors for the new shape
    x_scale = 1280 / old_width
    y_scale = 720 / old_height

    # Adjust the coordinates of the bounding box
    new_x1 = int(x1 * x_scale)
    new_y1 = int(y1 * y_scale)
    new_x2 = int(x2 * x_scale)
    new_y2 = int(y2 * y_scale)

    # Return the adjusted coordinates as a tuple
    return (new_x1, new_y1, new_x2, new_y2)
