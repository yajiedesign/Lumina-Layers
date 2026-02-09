"""
Lumina Studio - LUT Merger Utility

Merge multiple LUT files from different color modes to expand color gamut.
"""

import numpy as np
from scipy.spatial import KDTree


def merge_luts(primary_lut_path, secondary_lut_paths, min_distance=0.0):
    """
    Merge multiple LUT files, removing duplicate colors.
    
    Args:
        primary_lut_path: Path to primary LUT file (e.g., 8-color)
        secondary_lut_paths: List of paths to secondary LUT files (e.g., [6-color, 4-color, BW])
        min_distance: Minimum RGB distance to consider colors as different (default: 0.0, allows overlap)
    
    Returns:
        Tuple of (merged_lut_dict, stats_dict)
        merged_lut_dict contains:
            - 'colors': RGB array (N, 3)
            - 'stacks': Stacking array (N, 5) -堆叠信息
    """
    print("[LUT_MERGER] Starting LUT merge process...")
    
    # Load primary LUT with stacking info
    try:
        primary_lut = np.load(primary_lut_path)
        primary_colors = primary_lut.reshape(-1, 3)
        
        # Reconstruct primary stacking info
        primary_stacks = _reconstruct_stacks(primary_colors, primary_lut_path)
        
        print(f"[LUT_MERGER] Primary LUT loaded: {len(primary_colors)} colors from {primary_lut_path}")
    except Exception as e:
        raise ValueError(f"Failed to load primary LUT: {e}")
    
    # Initialize merged data
    merged_colors = primary_colors.tolist()
    merged_stacks = primary_stacks.tolist()
    
    stats = {
        'primary_count': len(primary_colors),
        'added_from_secondary': {},
        'duplicates_removed': 0,
        'total_merged': 0
    }
    
    # Build KD-Tree for fast nearest neighbor search
    kdtree = KDTree(primary_colors)
    
    # Process each secondary LUT
    for secondary_path in secondary_lut_paths:
        try:
            secondary_lut = np.load(secondary_path)
            secondary_colors = secondary_lut.reshape(-1, 3)
            
            # Reconstruct secondary stacking info
            secondary_stacks = _reconstruct_stacks(secondary_colors, secondary_path)
            
            print(f"[LUT_MERGER] Processing secondary LUT: {len(secondary_colors)} colors from {secondary_path}")
            
            added_count = 0
            duplicate_count = 0
            
            for i, color in enumerate(secondary_colors):
                # Find nearest color in merged set
                dist, _ = kdtree.query(color)
                
                if dist >= min_distance:
                    # Color is sufficiently different, add it with its stacking info
                    merged_colors.append(color)
                    merged_stacks.append(secondary_stacks[i])
                    added_count += 1
                else:
                    # Color is too similar, skip it
                    duplicate_count += 1
            
            # Rebuild KD-Tree with new colors
            if added_count > 0:
                kdtree = KDTree(np.array(merged_colors))
            
            stats['added_from_secondary'][secondary_path] = added_count
            stats['duplicates_removed'] += duplicate_count
            
            print(f"[LUT_MERGER]   Added: {added_count}, Duplicates: {duplicate_count}")
            
        except Exception as e:
            print(f"[LUT_MERGER] Warning: Failed to load {secondary_path}: {e}")
            continue
    
    # Convert to numpy arrays
    merged_colors_array = np.array(merged_colors, dtype=np.uint8)
    merged_stacks_array = np.array(merged_stacks, dtype=np.int8)
    
    # Ensure correct shapes
    if merged_colors_array.ndim == 1:
        merged_colors_array = merged_colors_array.reshape(-1, 3)
    if merged_stacks_array.ndim == 1:
        merged_stacks_array = merged_stacks_array.reshape(-1, 5)
    
    stats['total_merged'] = len(merged_colors_array)
    
    print(f"[LUT_MERGER] ✅ Merge complete: {stats['total_merged']} total colors")
    
    # Return both colors and stacks
    merged_lut_dict = {
        'colors': merged_colors_array,
        'stacks': merged_stacks_array
    }
    
    return merged_lut_dict, stats


def _reconstruct_stacks(colors, lut_path):
    """
    Reconstruct stacking information from LUT colors.
    
    Args:
        colors: RGB color array (N, 3)
        lut_path: Path to LUT file (for detecting mode)
    
    Returns:
        Stacking array (N, 5)
    """
    total_colors = len(colors)
    
    # Detect LUT type based on color count
    if total_colors == 32:
        # BW mode: 2^5 = 32
        print(f"[LUT_MERGER]   Reconstructing BW stacks (2^5)")
        stacks = []
        for i in range(32):
            digits = []
            temp = i
            for _ in range(5):
                digits.append(temp % 2)
                temp //= 2
            stack = digits[::-1]  # [顶...底] format
            stacks.append(stack)
        return np.array(stacks, dtype=np.int8)
    
    elif total_colors == 1024:
        # 4-color mode: 4^5 = 1024
        print(f"[LUT_MERGER]   Reconstructing 4-color stacks (4^5)")
        stacks = []
        for i in range(1024):
            digits = []
            temp = i
            for _ in range(5):
                digits.append(temp % 4)
                temp //= 4
            stack = digits[::-1]  # [顶...底] format
            stacks.append(stack)
        return np.array(stacks, dtype=np.int8)
    
    elif 1200 <= total_colors <= 1400:
        # 6-color mode: Smart 1296
        print(f"[LUT_MERGER]   Reconstructing 6-color stacks (Smart 1296)")
        from core.calibration import get_top_1296_colors
        smart_stacks = get_top_1296_colors()
        # Reverse for Face-Down printing
        smart_stacks = [tuple(reversed(s)) for s in smart_stacks]
        return np.array(smart_stacks[:total_colors], dtype=np.int8)
    
    elif 2600 <= total_colors <= 2800:
        # 8-color mode: Smart 2738
        print(f"[LUT_MERGER]   Reconstructing 8-color stacks (Smart 2738)")
        smart_stacks = np.load('assets/smart_8color_stacks.npy')
        # Reverse for Face-Down printing
        smart_stacks = np.array([s[::-1] for s in smart_stacks])
        return smart_stacks[:total_colors].astype(np.int8)
    
    else:
        # Unknown format - use intelligent color analysis
        print(f"[LUT_MERGER]   Warning: Unknown LUT format ({total_colors} colors), using color analysis")
        return _analyze_color_stacks(colors)


def _analyze_color_stacks(colors):
    """
    Analyze colors and infer stacking information.
    Fallback method for unknown LUT formats.
    
    Args:
        colors: RGB color array (N, 3)
    
    Returns:
        Stacking array (N, 5)
    """
    # Detect number of materials based on unique color components
    # This is a heuristic approach
    
    # Try to detect if it's 6-color or 8-color based on color diversity
    unique_r = len(np.unique(colors[:, 0]))
    unique_g = len(np.unique(colors[:, 1]))
    unique_b = len(np.unique(colors[:, 2]))
    
    avg_unique = (unique_r + unique_g + unique_b) / 3
    
    if avg_unique > 100:
        # Likely 8-color
        num_materials = 8
        base_colors = np.array([
            [255, 255, 255],  # 0: White
            [0, 134, 214],    # 1: Cyan
            [236, 0, 140],    # 2: Magenta
            [244, 238, 42],   # 3: Yellow
            [20, 20, 20],     # 4: Black
            [193, 46, 31],    # 5: Red
            [10, 41, 137],    # 6: Deep Blue
            [0, 174, 66]      # 7: Green
        ], dtype=float)
    else:
        # Likely 6-color
        num_materials = 6
        base_colors = np.array([
            [255, 255, 255],  # 0: White
            [0, 134, 214],    # 1: Cyan
            [236, 0, 140],    # 2: Magenta
            [0, 174, 66],     # 3: Green
            [244, 238, 42],   # 4: Yellow
            [20, 20, 20]      # 5: Black
        ], dtype=float)
    
    print(f"[LUT_MERGER]   Analyzing colors: detected {num_materials}-material system")
    
    # For each color, find closest material and use it for all layers
    stacks = []
    for color in colors:
        distances = np.linalg.norm(base_colors - color.astype(float), axis=1)
        closest_mat = np.argmin(distances)
        stack = [int(closest_mat)] * 5
        stacks.append(stack)
    
    return np.array(stacks, dtype=np.int8)


def validate_lut_compatibility(lut_path):
    """
    Validate LUT file and detect its color mode.
    
    Args:
        lut_path: Path to LUT file (.npy or .npz)
    
    Returns:
        Tuple of (is_valid, color_count, detected_mode)
    """
    try:
        # Check if this is a merged LUT (.npz format)
        if lut_path.endswith('.npz'):
            lut_data = np.load(lut_path)
            colors = lut_data['colors']
            color_count = len(colors)
        else:
            # Standard .npy format
            lut_data = np.load(lut_path)
            colors = lut_data.reshape(-1, 3)
            color_count = len(colors)
        
        # Detect mode based on color count
        if 30 <= color_count <= 35:
            detected_mode = "BW (2-Color)"
        elif 900 <= color_count <= 1100:
            detected_mode = "4-Color (CMYW/RYBW)"
        elif 1200 <= color_count <= 1400:
            detected_mode = "6-Color Smart"
        elif 2600 <= color_count <= 2800:
            detected_mode = "8-Color Max"
        else:
            detected_mode = f"Unknown ({color_count} colors)"
        
        return True, color_count, detected_mode
        
    except Exception as e:
        return False, 0, f"Error: {e}"


def get_merge_recommendations(primary_mode):
    """
    Get recommended secondary LUTs based on primary mode.
    
    Args:
        primary_mode: Primary LUT mode string
    
    Returns:
        List of recommended secondary modes
    """
    recommendations = {
        "8-Color Max": ["6-Color Smart", "4-Color (CMYW/RYBW)", "BW (2-Color)"],
        "6-Color Smart": ["4-Color (CMYW/RYBW)", "BW (2-Color)"],
        "4-Color (CMYW/RYBW)": ["BW (2-Color)"],
        "BW (2-Color)": []
    }
    
    return recommendations.get(primary_mode, [])
