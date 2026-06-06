from fake_ripple_generator import generate_synthetic_lfp
from scipy.signal import decimate

from espresso.hfo.ripple_detector import detect_ripples
from espresso.models.ripple_dataset import RippleDataset
from espresso.models.ripple_event import RippleEvent
from espresso.ui.ripple_viewer import RippleViewer
from espresso.ui.state.ripple_viewer_controller import RippleViewerController
from espresso.ui.state.ripple_viewer_state import RippleViewerParams


def run_ripple_analysis() -> None:
    fs_raw = 32000
    duration_s = 500.0

    timestamps_raw, signal_raw, ripples = generate_synthetic_lfp(fs_raw, duration_s, 20)

    data_8khz = decimate(signal_raw, q=4, ftype="iir", zero_phase=True)
    data_2khz = decimate(data_8khz, q=4, ftype="iir", zero_phase=True)

    timestamps_ds = timestamps_raw[::16]

    min_len = min(len(timestamps_ds), len(data_2khz))
    timestamps_ds = timestamps_ds[:min_len]
    data_2khz = data_2khz[:min_len]

    events1: list[RippleEvent] = detect_ripples(
        time=timestamps_ds,
        signals=data_2khz,
        threshold_dev=[3, 100],
    )

    events2: list[RippleEvent] = detect_ripples(
        time=timestamps_ds,
        signals=data_2khz,
        threshold_dev=[1.5, 8],
    )

    duration_m = duration_s // 60
    print(f"File Duration: {duration_m:.0f}m {duration_s % 60:.2f}s")
    print("Detected Ripple Peaks:")
    for peak in events1[:5]:
        print(f"{peak}\n")

    ripple_datasets = [
        RippleDataset(
            label="raw",
            raw_microvolts={"channel_0": signal_raw},
            ripples={"channel_0": events1},
            fs=fs_raw,
        ),
        RippleDataset(
            label="another",
            raw_microvolts={"channel_0": data_2khz},
            ripples={"channel_0": events2},
            fs=2000,
        ),
    ]

    viewer_params = RippleViewerParams()

    viewer_controller = RippleViewerController(
        ripple_datasets=ripple_datasets,
        ripple_viewer_params=viewer_params,
    )
    viewer = RippleViewer(
        controller=viewer_controller,
    )
    viewer.run()


if __name__ == "__main__":
    run_ripple_analysis()
