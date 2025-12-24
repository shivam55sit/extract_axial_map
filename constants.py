"""
Constants for Pentacam/Oculyzer image processing
Focus: Axial map extraction and cleaning
"""

import numpy as np

# Supported image dimensions
SUPPORTED_DIMENSIONS = [
    '740_1200', '758_1200', '820_1200', '838_1200', '840_1200',
    '858_1200', '894_1600', '904_1200', '910_1200', '940_1200'
]

# Header extraction coordinates (top section of images)
IMAGE_HEADER_COORDS = {
    '758_1200': [2, 40, 2, 948],
    '820_1200': [2, 40, 2, 948],
    '840_1200': [2, 40, 2, 948],
    '894_1600': [2, 40, 2, 948],
    '740_1200': [2, 40, 2, 948],
    '838_1200': [2, 40, 2, 948],
    '858_1200': [100, 140, 2, 948],
    '904_1200': [70, 110, 2, 948],
    '910_1200': [2, 40, 2, 948],
    '940_1200': [100, 140, 2, 948]
}

# Expected header texts
MODEL_HEADERS = [
    'WAVELIGHT ALLEGRO OCULYZER 4 Maps Refractive',
    'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable',
    'OCULUS PENTACAM 4 Maps Refractive',
    'OCULUS PENTACAM 4 Maps Selectable'
]

OCULYZER_HEADERS = [
    'WAVELIGHT ALLEGRO OCULYZER 4 Maps Refractive',
    'WAVELIGHT ALLEGRO OCULYZER 4 Maps Selectable'
]

PENTACAM_HEADERS = [
    'OCULUS PENTACAM 4 Maps Refractive',
    'OCULUS PENTACAM 4 Maps Selectable'
]

# Dimension string to tuple mapping
IMG_SIZE_TO_DIMENSIONS = {
    '740_1200': (740, 1200),
    '758_1200': (758, 1200),
    '820_1200': (820, 1200),
    '840_1200': (840, 1200),
    '894_1600': (894, 1600),
    '838_1200': (838, 1200),
    '858_1200': (858, 1200),
    '904_1200': (904, 1200),
    '910_1200': (910, 1200),
    '940_1200': (940, 1200)
}

# Axial map coordinates for REFRACTIVE images
# Format: {dimension: {'map1': coords for top-left axial map, ...}}
REFRACTIVE_MAP_COORDS = {
    (740, 1200): {
        'map1': {'row1': 110, 'row2': 340, 'col1': 420, 'col2': 650},
        'circle_loc': {'cir_x': 115, 'cir_y': 115, 'cir_radius': 111}
    },
    (758, 1200): {
        'map1': {'row1': 115, 'row2': 387, 'col1': 431, 'col2': 703},
        'circle_loc': {'cir_x': 136, 'cir_y': 136, 'cir_radius': 126}
    },
    (820, 1200): {
        'map1': {'row1': 124, 'row2': 416, 'col1': 441, 'col2': 733},
        'circle_loc': {'cir_x': 146, 'cir_y': 146, 'cir_radius': 138}
    },
    (838, 1200): {
        'map1': {'row1': 128, 'row2': 424, 'col1': 447, 'col2': 741},
        'circle_loc': {'cir_x': 146, 'cir_y': 148, 'cir_radius': 140}
    },
    (840, 1200): {
        'map1': {'row1': 128, 'row2': 424, 'col1': 447, 'col2': 743},
        'circle_loc': {'cir_x': 148, 'cir_y': 148, 'cir_radius': 140}
    },
    (894, 1600): {
        'map1': {'row1': 132, 'row2': 448, 'col1': 559, 'col2': 875},
        'circle_loc': {'cir_x': 158, 'cir_y': 158, 'cir_radius': 150}
    },
    (858, 1200): {
        'map1': {'row1': 220, 'row2': 484, 'col1': 433, 'col2': 697},
        'circle_loc': {'cir_x': 132, 'cir_y': 132, 'cir_radius': 126}
    },
    (904, 1200): {
        'map1': {'row1': 190, 'row2': 490, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 150, 'cir_radius': 140}
    },
    (910, 1200): {
        'map1': {'row1': 124, 'row2': 424, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 150, 'cir_radius': 140}
    },
    (940, 1200): {
        'map1': {'row1': 230, 'row2': 526, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 147, 'cir_radius': 140}
    }
}

# Axial map coordinates for SELECTABLE images
SELECTABLE_MAP_COORDS = {
    (758, 1200): {
        'map1': {'row1': 115, 'row2': 387, 'col1': 431, 'col2': 703},
        'circle_loc': {'cir_x': 136, 'cir_y': 136, 'cir_radius': 126}
    },
    (820, 1200): {
        'map1': {'row1': 94, 'row2': 386, 'col1': 441, 'col2': 733},
        'circle_loc': {'cir_x': 146, 'cir_y': 146, 'cir_radius': 142}
    },
    (838, 1200): {
        'map1': {'row1': 128, 'row2': 424, 'col1': 447, 'col2': 741},
        'circle_loc': {'cir_x': 146, 'cir_y': 148, 'cir_radius': 140}
    },
    (840, 1200): {
        'map1': {'row1': 94, 'row2': 390, 'col1': 447, 'col2': 743},
        'circle_loc': {'cir_x': 148, 'cir_y': 148, 'cir_radius': 140}
    },
    (894, 1600): {
        'map1': {'row1': 93, 'row2': 419, 'col1': 554, 'col2': 880},
        'circle_loc': {'cir_x': 164, 'cir_y': 163, 'cir_radius': 158}
    },
    (858, 1200): {
        'map1': {'row1': 220, 'row2': 484, 'col1': 433, 'col2': 697},
        'circle_loc': {'cir_x': 132, 'cir_y': 132, 'cir_radius': 126}
    },
    (904, 1200): {
        'map1': {'row1': 190, 'row2': 490, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 150, 'cir_radius': 140}
    },
    (910, 1200): {
        'map1': {'row1': 94, 'row2': 394, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 150, 'cir_radius': 140}
    },
    (940, 1200): {
        'map1': {'row1': 230, 'row2': 526, 'col1': 445, 'col2': 741},
        'circle_loc': {'cir_x': 147, 'cir_y': 147, 'cir_radius': 140}
    }
}

# Standard output size for processed images
OUTPUT_SIZE = 224