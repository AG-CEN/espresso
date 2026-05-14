import numpy as np
from scipy.signal import decimate

from espresso.hfo.ripple_detector import detect_ripples
from espresso.ui.ripple_viewer import RippleEvent, RippleViewer


def run_ripple_analysis() -> None:
    fs_original = 32000.0
    duration_seconds = 10.0
    total_samples = int(fs_original * duration_seconds)

    # Simulated background activity with theta-band oscillation.
    raw_signal = np.random.normal(0, 0.1, total_samples)
    theta_wave = (
        np.sin(2 * np.pi * 8 * np.linspace(0, duration_seconds, total_samples)) * 0.5
    )
    raw_signal += theta_wave

    # Inject a synthetic ripple-like burst.
    burst_start = int(4.0 * fs_original)
    burst_end = int(4.2 * fs_original)
    t_burst = np.linspace(0, 0.2, burst_end - burst_start)
    ripple_burst = (
        np.sin(2 * np.pi * 200 * t_burst) * np.hanning(burst_end - burst_start) * 4.0
    )
    raw_signal[burst_start:burst_end] += ripple_burst

    fs = fs_original
    epoch_time = len(raw_signal) / fs
    epoch_min = epoch_time // 60

    timestamps_seconds = np.linspace(0, duration_seconds, total_samples)

    # Two-stage decimation: 32 kHz -> 8 kHz -> 2 kHz.
    data_8khz = decimate(raw_signal, q=4, ftype='iir', zero_phase=True)
    data_2khz = decimate(data_8khz, q=4, ftype='iir', zero_phase=True)

    timestamps_seconds_ds = timestamps_seconds[::16]

    # Align arrays after decimation edge effects.
    if len(timestamps_seconds_ds) > len(data_2khz):
        timestamps_seconds_ds = timestamps_seconds_ds[: len(data_2khz)]
    elif len(data_2khz) > len(timestamps_seconds_ds):
        data_2khz = data_2khz[: len(timestamps_seconds_ds)]

    assert len(timestamps_seconds_ds) == len(data_2khz), (
        'Time and signal lengths mismatch.'
    )

    (ripple_peak_time, _), ripple_segments, (low, high) = detect_ripples(
        time=timestamps_seconds_ds,
        signals=data_2khz,
        threshold_dev=[3, 6],
    )

    print(f'File Duration: {epoch_min}m {epoch_time % 60:.2f}s')
    print(f'Detected Ripple Peaks: {ripple_peak_time}')

    events: list[RippleEvent] = [
        RippleEvent(start_sec=float(s), end_sec=float(e), peak_sec=float(p))
        for s, e, p in zip(
            ripple_segments.start,
            ripple_segments.stop,
            ripple_peak_time,
            strict=True,
        )
    ]

    view = RippleViewer(
        raw_volts={'channel_0': raw_signal},
        ripples={'channel_0': events},
        fs=fs,
    )
    view.showMaximized()


if __name__ == '__main__':
    run_ripple_analysis()
