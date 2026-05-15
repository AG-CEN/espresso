# espresso

A Python framework for preprocessing, processing, visualization, event detection, and event curation of high-density time-series signals and multi-channel data streams.

### Key Capabilities
* **Signal Preprocessing:** Optimized digital filtering pipelines and artifact rejection workflows.
* **Event Detection:** Automated extraction of transient oscillations and sharp-wave ripples.
* **Signal Visualization:** Modern hardware-accelerated time-series traces and interpolated spectrogram displays.
* **Event Curation:** Interactive manual and automated classification workflows to review, filter, and tag detected transient neural events. (Coming soon)


## Installation

```bash
pip install espresso-neuro
```

## Usage

Refer to the [examples/](examples/) directory for complete, runnable pipeline scripts demonstrating signal processing, downsampling metrics, and hardware-accelerated user interface execution.

## License & Attribution

This project is licensed under the GNU General Public License v3 - see the [LICENSE](LICENSE) file for details.

### Third-Party Code
* The ripple detection module in `src/espresso/ripple_detector/` contains algorithm logic adapted from the [FKLab Python Core library](https://bitbucket.org/kloostermannerflab/fklab-python-core) by the Kloosterman Lab.
