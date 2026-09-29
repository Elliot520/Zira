"""Command schema sent from the JARVIS server to the Raspberry Pi robot."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_MOVEMENT_SECONDS = 10.0


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    CURIOUS = "curious"
    SURPRISED = "surprised"
    THINKING = "thinking"


class Head(str, Enum):
    NONE = "none"
    NOD = "nod"
    SHAKE = "shake"
    TILT_LEFT = "tilt_left"
    TILT_RIGHT = "tilt_right"
    UP = "up"
    DOWN = "down"


class Eyes(str, Enum):
    IDLE = "idle"
    LOOK_AT_USER = "look_at_user"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    BLINK = "blink"


class Gesture(str, Enum):
    NONE = "none"
    WAVE = "wave"
    POINT = "point"
    SHRUG = "shrug"
    THUMBS_UP = "thumbs_up"


class Movement(str, Enum):
    STOP = "stop"
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class RobotCommand(BaseModel):
    """One expressive action: speech + face/head/gesture + optional bounded movement.

    Safety: any movement other than `stop` must have a duration, capped at
    MAX_MOVEMENT_SECONDS, so a command can never request unbounded motion.
    """

    model_config = ConfigDict(extra="forbid")

    speech: str | None = Field(default=None, max_length=1000)
    emotion: Emotion = Emotion.NEUTRAL
    head: Head = Head.NONE
    eyes: Eyes = Eyes.IDLE
    gesture: Gesture = Gesture.NONE
    movement: Movement = Movement.STOP
    duration: float = Field(default=0.0, ge=0.0, le=MAX_MOVEMENT_SECONDS)

    @model_validator(mode="after")
    def movement_needs_duration(self) -> "RobotCommand":
        if self.movement is not Movement.STOP and self.duration <= 0:
            raise ValueError("movement requires a duration greater than 0")
        return self
