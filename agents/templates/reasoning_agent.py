import base64
import io
import json
import logging
import textwrap
from typing import Any, Dict, List, Literal

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..structs import FrameData, GameAction
from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)


class ReasoningActionResponse(BaseModel):
    """Action response structure for reasoning agent."""

    name: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"] = Field(
        description="The action to take."
    )
    reason: str = Field(
        description="Detailed reasoning for choosing this action",
        min_length=10,
        max_length=2000,
    )
    short_description: str = Field(
        description="Brief description of the action", min_length=5, max_length=500
    )
    hypothesis: str = Field(
        description="Current hypothesis about game mechanics",
        min_length=10,
        max_length=2000,
    )
    aggregated_findings: str = Field(
        description="Summary of discoveries and learnings so far",
        min_length=10,
        max_length=2000,
    )


class ReasoningAgent(ReasoningLLM):
    """A reasoning agent that tracks screen history and builds hypotheses about game rules."""

    MAX_ACTIONS = 400
    DO_OBSERVATION = True
    MODEL = "gpt-5"
    MESSAGE_LIMIT = 5
    REASONING_EFFORT = "high"
    ZONE_SIZE = 16

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: List[ReasoningActionResponse] = []
        self.screen_history: List[bytes] = []
        self.max_screen_history = 10  # Limit screen history to prevent memory leak
        self.client = OpenAI()

    def clear_history(self) -> None:
        """Clear all history when transitioning between levels."""
        self.history = []
        self.screen_history = []

    def generate_grid_image_with_zone(
        self, grid: List[List[int]], cell_size: int = 40
    ) -> bytes:
        """Generate PIL image of the grid with colored cells and zone coordinates."""
        if not grid or not grid[0]:
            # Create empty image
            img = Image.new("RGB", (200, 200), color="black")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

        height = len(grid)
        width = len(grid[0])

        # Create image
        img = Image.new("RGB", (width * cell_size, height * cell_size), color="white")
        draw = ImageDraw.Draw(img)

        # Color mapping for grid cells
        key_colors = {
            0: "#FFFFFF",
            1: "#CCCCCC",
            2: "#999999",
            3: "#666666",
            4: "#333333",
            5: "#000000",
            6: "#E53AA3",
            7: "#FF7BCC",
            8: "#F93C31",
            9: "#1E93FF",
            10: "#88D8F1",
            11: "#FFDC00",
            12: "#FF851B",
            13: "#921231",
            14: "#4FCC30",
            15: "#A356D6",
        }

        # Draw grid cells
        for y in range(height):
            for x in range(width):
                color = key_colors.get(grid[y][x], "#888888")  # default: floor

                # Draw cell
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        (x + 1) * cell_size,
                        (y + 1) * cell_size,
                    ],
                    fill=color,
                    outline="#000000",
                    width=1,
                )

        # Draw zone coordinates and borders
        for y in range(0, height, self.ZONE_SIZE):
            for x in range(0, width, self.ZONE_SIZE):
                # Draw zone coordinate label
                try:
                    font = ImageFont.load_default()
                    zone_text = f"({x},{y})"
                    draw.text(
                        (x * cell_size + 2, y * cell_size + 2),
                        zone_text,
                        fill="#FFFFFF",
                        font=font,
                    )
                except (ImportError, OSError) as e:
                    logger.debug(f"Could not load font for zone labels: {e}")
                except Exception as e:
                    logger.error(f"Failed to draw zone label at ({x},{y}): {e}")

                # Draw zone boundary
                zone_width = min(self.ZONE_SIZE, width - x) * cell_size
                zone_height = min(self.ZONE_SIZE, height - y) * cell_size
                draw.rectangle(
                    [
                        x * cell_size,
                        y * cell_size,
                        x * cell_size + zone_width,
                        y * cell_size + zone_height,
                    ],
                    fill=None,
                    outline="#FFD700",  # gold border for zone
                    width=2,
                )

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def build_tools(self) -> list[dict[str, Any]]:
        """Build Responses-native tool descriptors with ReasoningActionResponse schema."""
        schema = ReasoningActionResponse.model_json_schema()
        schema["properties"].pop("name", None)
        if "required" in schema and "name" in schema["required"]:
            schema["required"].remove("name")
        return [
            {
                "type": "function",
                "name": action.name,
                "description": f"Take action {action.name}",
                "parameters": schema,
            }
            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.RESET,
            ]
        ]

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for hypothesis-driven exploration."""
        return textwrap.dedent(
            """
You are playing a video game.

Your ultimate goal is to understand the rules of the game and explain them to your colleagues.

The game is complex, and may look like an IQ test.

You need to determine how the game works on your own.

To do so, we will provide you with a view of the game corresponding to the bird-eye view of the game, along with the raw grid data.

You can do 5 actions:
- RESET (used to start a new game or level)
- ACTION1 (MOVE_UP)
- ACTION2 (MOVE_DOWN)
- ACTION3 (MOVE_LEFT)
- ACTION4 (MOVE_RIGHT)

You can do one action at once.

Every time an action is performed we will provide you with the previous screen and the current screen.

Determine the game rules based on how the game reacted to the previous action (based on the previous screen and the current screen).

Your goal:

1. Experiment the game to determine how it works based on the screens and your actions.
2. Analyse the impact of your actions by comparing the screens.

How to proceed:
1. Define an hypothesis and an action to validate it.
2. Once confirmed, store the findings. Summarize and aggregate them so that your colleagues can understand the game based on your learning.
3. Make sure to understand clearly the game rules, energy, walls, doors, keys, etc.

Hint:
- The game is a 2D platformer.
- The player can move up, down, left and right.
- The player has a blue body and an orange head.
- There are walls in black.
- The door has a pink border and a shape inside.
        """
        )

    def call_llm_with_structured_output(
        self, messages: List[Dict[str, Any]]
    ) -> ReasoningActionResponse:
        """Call LLM with structured output parsing for reasoning agent."""
        try:
            response = self.client.responses.create(
                model=self.MODEL,
                input=messages,
                tools=self.build_tools(),
                tool_choice="required",
                reasoning={"effort": getattr(self, "REASONING_EFFORT", None)}
                if getattr(self, "REASONING_EFFORT", None)
                else None,
            )

            # Token and reasoning tracking (Responses API)
            assistant_text = getattr(response, "output_text", None) or ""
            self.track_tokens(response.usage.total_tokens, assistant_text)
            self.capture_reasoning_from_response(response)

            # Find first function_call
            tool_call_item = None
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "function_call":
                    tool_call_item = item
                    break

            if tool_call_item is None:
                raise ValueError("LLM did not return a tool call.")

            arguments_json = getattr(tool_call_item, "arguments", "{}")
            name = getattr(tool_call_item, "name", "RESET")
            function_args = json.loads(arguments_json or "{}")
            function_args["name"] = name
            return ReasoningActionResponse(**function_args)

        except Exception as e:
            logger.error(f"LLM structured call failed: {e}")
            raise e

    def define_next_action(self, latest_frame: FrameData) -> ReasoningActionResponse:
        """Define next action for the reasoning agent."""
        # Generate map image
        current_grid = latest_frame.frame[-1] if latest_frame.frame else []
        map_image = self.generate_grid_image_with_zone(current_grid)

        # Build messages
        system_prompt = self.build_user_prompt(latest_frame)

        # Get latest action from history
        latest_action = self.history[-1] if self.history else None

        # Build user message with images
        user_message_content: List[Dict[str, Any]] = []

        # Use the last screen from history as the 'previous_screen'
        previous_screen = self.screen_history[-1] if self.screen_history else None

        if previous_screen:
            user_message_content.extend(
                [
                    {"type": "input_text", "text": "Previous screen:"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64.b64encode(previous_screen).decode()}",
                    },
                ]
            )

        raw_grid_text = self.pretty_print_3d(latest_frame.frame)
        user_message_text = f"Your previous action was: {json.dumps(latest_action.model_dump() if latest_action else None, indent=2)}\n\nAttached are the visual screen and raw grid data.\n\nRaw Grid:\n{raw_grid_text}\n\nWhat should you do next?"

        current_image_b64 = base64.b64encode(map_image).decode()
        user_message_content.extend(
            [
                {"type": "input_text", "text": user_message_text},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{current_image_b64}",
                },
            ]
        )

        # Build messages
        messages: List[Dict[str, Any]] = [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": user_message_content},
        ]

        # Call LLM with structured output
        result = self.call_llm_with_structured_output(messages)

        # Store current screen for next iteration (after using it)
        self.screen_history.append(map_image)
        if len(self.screen_history) > self.max_screen_history:
            self.screen_history.pop(0)

        return result

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using parent class tool calling with reasoning enhancement."""
        if latest_frame.full_reset:
            self.clear_history()
            return GameAction.RESET

        if not self.history:  # First action must be RESET
            action = GameAction.RESET
            initial_response = ReasoningActionResponse(
                name="RESET",
                reason="Initial action to start the game and observe the environment.",
                short_description="Start game",
                hypothesis="The game requires a RESET to begin.",
                aggregated_findings="No findings yet.",
            )
            self.history.append(initial_response)
            return action

        # Define the next action based on reasoning
        action_response = self.define_next_action(latest_frame)
        self.history.append(action_response)

        # Map the reasoning action name to a GameAction
        action = GameAction.from_name(action_response.name)

        # Create and attach reasoning metadata
        reasoning_meta = {
            "model": self.MODEL,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "agent_type": "reasoning_agent",
            "hypothesis": action_response.hypothesis,
            "aggregated_findings": action_response.aggregated_findings,
            "response_preview": action_response.reason[:200] + "..."
            if len(action_response.reason) > 200
            else action_response.reason,
            "action_chosen": action.name,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
        }
        action.reasoning = reasoning_meta

        return action
