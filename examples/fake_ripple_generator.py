import numpy as np


def generate_synthetic_lfp(
    fs: float, duration: float, num_ripples: int = 20
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    (
        """Generate a synthetic LFP array with ripples scaled to cross """
        """specific SD detection thresholds."""
    )
    total_samples = int(fs * duration)
    time_axis = np.linspace(0, duration, total_samples)

    # Base noise floor standard deviation (sigma = 15.0 uV)
    noise_std = 15.0
    noise = np.random.normal(0, noise_std, total_samples)
    theta_oscillation = np.sin(2 * np.pi * 8 * time_axis) * 150.0  # 150 uV carrier wave
    signal = noise + theta_oscillation

    # Evenly distribute ripples across the duration (preventing overlaps)
    start_buffer = 0.5
    end_buffer = duration - 0.5
    valid_timestamps = np.linspace(start_buffer, end_buffer, num_ripples)

    ripple_records = []
    for start_time in valid_timestamps:
        # Pass detection params to ensure the ripple is large enough to be detected
        metadata = _inject_random_ripple(
            signal=signal,
            fs=fs,
            start_time=start_time,
            noise_std=noise_std,
            threshold_dev=[3, 6],  # [Boundary Threshold, Peak Threshold]
        )
        ripple_records.append(metadata)

    return time_axis, signal, ripple_records


def _inject_random_ripple(
    signal: np.ndarray,
    fs: float,
    start_time: float,
    noise_std: float,
    threshold_dev: list[float],
) -> dict:
    (
        """Synthesize a randomized rat-band ripple designed to breach """
        """Z-score detection criteria."""
    )
    duration = np.random.uniform(0.08, 0.18)  # 80-180ms

    # Strictly bound frequencies inside the rat ripple band [140.0, 225.0] Hz
    # Allow a few Hz safety margin for down-drift near the tail
    base_frequency = np.random.uniform(170.0, 220.0)
    drift_profile = np.linspace(0, np.random.uniform(-15.0, -5.0), int(duration * fs))
    dynamic_frequency = base_frequency + drift_profile
    dynamic_frequency = np.clip(dynamic_frequency, 140.0, 225.0)

    # Peak detection requires crossing the upper threshold (6 SD)
    # Scale peak amplitude randomly between 6.5 and 10.0 SD of the
    # noise floor to ensure detection
    peak_sd_multiplier = np.random.uniform(threshold_dev[1] + 0.5, 10.0)
    amplitude = noise_std * peak_sd_multiplier

    start_idx = int(start_time * fs)
    end_idx = start_idx + len(dynamic_frequency)
    t_burst = np.linspace(0, duration, len(dynamic_frequency))

    # Asymmetric beta envelope (skewed peak)
    peak_center = np.random.uniform(0.35, 0.50)
    alpha = peak_center * 8.0
    beta_param = (1.0 - peak_center) * 8.0
    envelope = (t_burst / duration) ** (alpha - 1) * (1 - t_burst / duration) ** (
        beta_param - 1
    )
    if np.max(envelope) > 0:
        envelope /= np.max(envelope)

    phase_jitter = np.random.uniform(0, 2 * np.pi)

    ripple_burst = (
        np.sin(2 * np.pi * dynamic_frequency * t_burst + phase_jitter)
        * envelope
        * amplitude
    )

    signal[start_idx:end_idx] += ripple_burst

    return {
        "start_sec": start_time,
        "end_sec": start_time + duration,
        "peak_frequency": base_frequency,
        "amplitude_uv": amplitude,
        "z_score_peak": peak_sd_multiplier,
    }
