import numpy as np
from scipy.signal import decimate

from espresso.hfo.ripple_detector import detect_ripples
from espresso.models.ripple_event import RippleEvent
from espresso.ui.ripple_viewer import RippleViewer
from espresso.ui.ripple_viewer_controller import RippleViewerController


def inject_ripple(
    signal: np.ndarray,
    fs: float,
    start_time: float,
    duration: float = 0.2,
    frequency: float = 200.0,
    amplitude: float = 4.0,
) -> None:
    start_idx = int(start_time * fs)
    end_idx = start_idx + int(duration * fs)
    total_samples = end_idx - start_idx

    t_burst = np.linspace(0, duration, total_samples)
    ripple_burst = (
        np.sin(2 * np.pi * frequency * t_burst) * np.hanning(total_samples) * amplitude
    )
    signal[start_idx:end_idx] += ripple_burst


def generate_synthetic_lfp(fs: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    total_samples = int(fs * duration)
    time_axis = np.linspace(0, duration, total_samples)

    noise = np.random.normal(0, 0.1, total_samples)
    theta_oscillation = np.sin(2 * np.pi * 8 * time_axis) * 0.5
    signal = noise + theta_oscillation

    ripple_protocols = [
        {"start_time": 2.0, "frequency": 180.0, "amplitude": 4.0},
        {"start_time": 4.5, "frequency": 200.0, "amplitude": 3.5},
        {"start_time": 7.1, "frequency": 220.0, "amplitude": 5.0},
    ]

    for protocol in ripple_protocols:
        inject_ripple(signal=signal, fs=fs, **protocol)

    return time_axis, signal


def run_ripple_analysis() -> None:
    fs_raw = 32000.0
    duration_s = 10.0

    timestamps_raw, signal_raw = generate_synthetic_lfp(fs_raw, duration_s)

    data_8khz = decimate(signal_raw, q=4, ftype="iir", zero_phase=True)
    data_2khz = decimate(data_8khz, q=4, ftype="iir", zero_phase=True)

    timestamps_ds = timestamps_raw[::16]

    min_len = min(len(timestamps_ds), len(data_2khz))
    timestamps_ds = timestamps_ds[:min_len]
    data_2khz = data_2khz[:min_len]

    events: list[RippleEvent] = detect_ripples(
        time=timestamps_ds,
        signals=data_2khz,
        threshold_dev=[3, 6],
    )

    duration_m = duration_s // 60
    print(f"File Duration: {duration_m:.0f}m {duration_s % 60:.2f}s")
    print("Detected Ripple Peaks:")
    for peak in events[:5]:
        print(peak)

    viewer_controller = RippleViewerController(
        raw_volts={"channel_0": signal_raw},
        ripples={"channel_0": events},
        fs=fs_raw,
    )
    viewer = RippleViewer(controller=viewer_controller)
    viewer.run()


if __name__ == "__main__":
    run_ripple_analysis()
