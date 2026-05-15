from pydantic import BaseModel, field_validator


class RippleEvent(BaseModel):
    """Data model representing a single detected high-frequency ripple event.

    Tracks concurrent timelines: relative seconds for UI plotting and absolute
    microseconds for DAQ hardware synchronization.
    """

    start_sec: float
    """Event onset in seconds relative to recording start (starts at 0.0)."""

    end_sec: float
    """Event offset in seconds relative to recording start."""

    peak_sec: float
    """Max amplitude position in seconds relative to recording start."""

    @field_validator('end_sec')
    @classmethod
    def end_must_be_after_start(cls, v: float, info):
        """Validates that the relative offset time occurs after the onset."""
        if 'start_sec' in info.data and v <= info.data['start_sec']:
            raise ValueError('end_sec must be greater than start_sec')
        return v
