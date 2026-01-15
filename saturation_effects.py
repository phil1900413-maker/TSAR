
import random
import numpy as np
import scipy.ndimage as ndi

def _generate_low_freq_noise(num_points, amplitude):
    """
    Вспомогательная функция для генерации "ломаной линии" (разреженных штрихов).
    """
    if num_points < 3 or amplitude < 1e-12:
        return np.zeros(num_points)
    
    num_strokes = np.random.randint(max(3, num_points // 20), max(4, num_points // 5))
    anchor_x = np.linspace(0, num_points - 1, num_strokes)
    anchor_y = np.random.uniform(-amplitude, amplitude, size=num_strokes)
    interp_x = np.arange(num_points)
    
    return np.interp(interp_x, anchor_x, anchor_y)

def _create_crater_artifact(cutoff_region, params):
    """
    Создает артефакт "Кратер" для ШИРОКИХ пиков.
    Логика взята из старой функции apply_crater_saturation.
    """
    offset_factor = np.random.uniform(params['crater_min_depth'], params['crater_max_bulge'])
    base_offset = offset_factor * cutoff_region
    
    jaggedness_amplitude_factor = np.random.uniform(0, params['jaggedness'])
    jaggedness_amplitude = jaggedness_amplitude_factor * np.mean(cutoff_region)
    jagged_noise = _generate_low_freq_noise(len(cutoff_region), jaggedness_amplitude)

    crater_values = cutoff_region + base_offset + jagged_noise
    return np.clip(crater_values, 0, None)

def _create_tower_artifact(cutoff_region, params):
    """
    Создает артефакт "Замок с башнями" для УЗКИХ пиков.
    НОВАЯ ВЕРСИЯ: Высота каждой башни выбирается случайно из диапазона.
    """
    n_pts = len(cutoff_region)
    tower_w = int(params.get('tower_width_pts', 4))

    if n_pts <= 2 * tower_w:
        return _create_crater_artifact(cutoff_region, params)

    mean_cutoff = np.mean(cutoff_region)
    
    
    moat_level = cutoff_region * params.get('moat_depth_factor', 0.85)
    moat_noise_amp = mean_cutoff * params.get('moat_noise_factor', 0.05)
    moat_noise = _generate_low_freq_noise(n_pts, moat_noise_amp)
    artifact = moat_level + moat_noise
    
    
    
    h_min, h_max = params.get('tower_height_range', (0.95, 1.15))
    
    
    random_factor_left = random.uniform(h_min, h_max)
    tower_h_left = mean_cutoff * random_factor_left
    
    
    random_factor_right = random.uniform(h_min, h_max)
    tower_h_right = mean_cutoff * random_factor_right

    
    left_tower_base = np.linspace(artifact[0], tower_h_left, tower_w)
    artifact[0:tower_w] = left_tower_base

    
    right_tower_base = np.linspace(tower_h_right, artifact[-1], tower_w)
    artifact[-tower_w:] = right_tower_base
    
    return np.clip(artifact, 0, None)


def apply_smart_saturation(y_signal, y_cutoff, x_axis, params):
    """
    "Умный" диспетчер. Анализирует ширину каждого насыщенного пика
    и применяет соответствующий артефакт ("Кратер" или "Замок с башнями").
    """
    is_saturated_mask = y_signal > y_cutoff
    if not np.any(is_saturated_mask):
        return y_signal

    labeled_regions, num_peaks = ndi.label(is_saturated_mask, structure=np.ones(3))
    y_modified = np.copy(y_signal)
    
    width_threshold = params.get('width_threshold', 50)

    for i in range(1, num_peaks + 1):
        region_mask = (labeled_regions == i)
        
        
        width = np.sum(region_mask)
        cutoff_in_region = y_cutoff[region_mask]
        
        if width < width_threshold:
            
            artifact_values = _create_tower_artifact(cutoff_in_region, params)
        else:
            
            artifact_values = _create_crater_artifact(cutoff_in_region, params)
            
        y_modified[region_mask] = artifact_values

    return y_modified
