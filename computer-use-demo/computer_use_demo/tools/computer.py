import asyncio
import base64
import io
import os
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
import tempfile

import pyautogui
from anthropic.types.beta import BetaToolComputerUse20241022Param
from mss import mss
from PIL import Image

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = Path(tempfile.gettempdir()) / "outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None
    # Choosing WXGA since it's close to Macbook Pro 14" scaling factor.
    target_width: int = 1280  # FWXGA width
    target_height: int = 800  # FWXGA height

    _screenshot_delay = 0.5
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()
        # Initialize pyautogui with a fail-safe
        pyautogui.FAILSAFE = True
        
        # Get the primary screen resolution
        self.width, self.height = pyautogui.size()
        self.display_num = 1  # Using primary display for now
        
        # Calculate scaling factors to convert FROM target TO actual screen resolution
        self.x_scaling_factor = self.width / self.target_width  # Now dividing width by target_width
        self.y_scaling_factor = self.height / self.target_height  # Now dividing height by target_height
        
        print(f"Screen resolution: {self.width}x{self.height}")
        print(f"Target resolution: {self.target_width}x{self.target_height}")
        print(f"Scaling factors: x={self.x_scaling_factor:.3f}, y={self.y_scaling_factor:.3f}")
        
        # Initialize mss for screenshots
        self.mss = mss()

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            if coordinate[0] > self.width or coordinate[1] > self.height:
                raise ToolError(f"Coordinates {coordinate[0]}, {coordinate[1]} are out of bounds")

            x, y = self.scale_coordinates(coordinate[0], coordinate[1])

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return ToolResult()
            elif action == "left_click_drag":
                current_x, current_y = pyautogui.position()
                pyautogui.mouseDown()
                pyautogui.moveTo(x, y)
                pyautogui.mouseUp()
                return ToolResult()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle Mac-specific key combinations
                if "Command" in text or "command" in text:
                    # Convert "Command_L+q" or "command+q" to ["command", "q"]
                    keys = text.lower().replace("_l", "").split("+")
                    pyautogui.hotkey(*keys)
                else:
                    pyautogui.press(text)
                return ToolResult()
            elif action == "type":
                pyautogui.write(text, interval=TYPING_DELAY_MS/1000)
                return await self.screenshot()

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                scaled_x, scaled_y = self.scale_coordinates(x, y)
                return ToolResult(output=f"X={scaled_x},Y={scaled_y}")
            else:
                click_functions = {
                    "left_click": pyautogui.click,
                    "right_click": pyautogui.rightClick,
                    "middle_click": pyautogui.middleClick,
                    "double_click": lambda: pyautogui.click(clicks=2),
                }
                click_functions[action]()
                return await self.screenshot()

        raise ToolError(f"Invalid action: {action}")

    def scale_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates FROM target resolution TO actual screen resolution."""
        if not self._scaling_enabled:
            return x, y
            
        scaled_x = round(x * self.x_scaling_factor)  # Now scaling up instead of down
        scaled_y = round(y * self.y_scaling_factor)
        
        print(f"Scaling coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
        return scaled_x, scaled_y

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        # Capture the primary monitor
        screenshot = self.mss.grab(self.mss.monitors[1])  # monitor 1 is the primary
        
        # Convert to PIL Image
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        print(f"Captured image resolution: {img.size[0]}x{img.size[1]}")
        
        if self._scaling_enabled:
            img = img.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            print(f"Scaled image to: {self.target_width}x{self.target_height}")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        
        return ToolResult(base64_image=base64_image)
