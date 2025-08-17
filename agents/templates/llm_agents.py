import base64
import io
import json
import logging
import os
from typing import Any, List, Optional

from langfuse.openai import OpenAI as OpenAIClient
from langfuse.openai import openai
from openai.types.responses import Response, ResponseInputParam, ResponseUsage
from PIL import Image, ImageDraw, ImageFont

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .prompts import (
    build_function_call_output_text,
    build_summarize_prompt,
    get_developer_prompt,
)

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 10
    REASONING_EFFORT: Optional[str] = "high"
    MODEL: str = "gpt-5"
    ZONE_SIZE: int = 16
    CELL_SIZE: int = 80
    ZONE_FONT_SIZE: int = 24
    REASONING_LOG_PREVIEW_CHARS: int = 2000
    token_counter: int
    current_thread_tokens: int
    client: OpenAIClient

    _latest_tool_call_id: str = "call_placeholder"

    # Compacting / handoff
    running_summary: str = ""
    MAX_SUMMARY_TOKENS: int = 20000
    TOKEN_SUMMARIZE_THRESHOLD: int = 200_000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        self.current_thread_tokens = 0
        self._prev_resp_id: Optional[str] = None
        self.observations: list[str] = []
        self._last_observations_turn: list[str] = []
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}"
        name += f".{self.REASONING_EFFORT}"
        return name

    def _responses_create(
        self,
        *,
        developer_text: Optional[str] = None,
        user_text: Optional[str] = None,
        function_call_output: Optional[tuple[str, str, str]] = None,
        image_base64: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Response:
        """Queries the LLM with the given parameters."""
        input_items: ResponseInputParam = []
        if function_call_output is not None:
            call_id, output_str, grid_base64 = function_call_output
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(output_str),
                }
            )
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{grid_base64}",
                        }
                    ],
                }
            )
        if user_text is not None:
            input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            )
        if image_base64 is not None:
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        }
                    ],
                }
            )
        create_kwargs: dict[str, Any] = {
            "model": self.MODEL,
            "input": input_items,
        }
        if developer_text is not None:
            create_kwargs["instructions"] = developer_text
        if tools is not None:
            create_kwargs["tools"] = tools
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice
        if previous_response_id is not None:
            create_kwargs["previous_response_id"] = previous_response_id
        if reasoning_effort is None:
            reasoning_effort = self.REASONING_EFFORT
        if reasoning_effort is not None:
            create_kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = max_output_tokens

        try:
            response = self.client.responses.create(**create_kwargs)
            return response
        except Exception:
            raise

    def generate_grid_image_with_zone(
        self, grid: List[List[int]], cell_size: int = 40
    ) -> str:
        """Generate a base64 PNG of the grid with colored cells and zone coordinates."""
        if not grid or not grid[0]:
            # Create empty image
            img = Image.new("RGB", (200, 200), color="black")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

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
                    font = ImageFont.load_default(size=self.ZONE_FONT_SIZE)
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

        # uncomment to save the image
        with open(f"images/grid_{self.action_counter:04d}.png", "wb") as f:
            f.write(buffer.getvalue())

        return base64.b64encode(buffer.getvalue()).decode()

    def _summarize(
        self,
        latest_frame: FrameData,
        function_response: tuple[str, str, str],
    ) -> None:
        """Use the LLM to write a concise handoff summary, then reset the thread."""
        try:
            prompt_text = build_summarize_prompt(
                self.pretty_print_3d(latest_frame.frame)
            )

            response = self._responses_create(
                developer_text=get_developer_prompt(),
                user_text=prompt_text,
                function_call_output=function_response,
                previous_response_id=self._prev_resp_id,
                max_output_tokens=self.MAX_SUMMARY_TOKENS,
            )

            assistant_text = ""
            reasoning_text = ""
            for item in response.output or []:
                if item.type == "message":
                    for elem in item.content or []:
                        if elem.type == "output_text" and elem.text:
                            assistant_text += elem.text or ""
                elif item.type == "reasoning":
                    for summary in item.summary:
                        reasoning_text += summary.text or ""

            self.running_summary = assistant_text
            self.track_tokens(response.usage, assistant_text, reasoning_text)

            if reasoning_text:
                preview = (
                    reasoning_text
                    if len(reasoning_text) <= self.REASONING_LOG_PREVIEW_CHARS
                    else reasoning_text[: self.REASONING_LOG_PREVIEW_CHARS] + "..."
                )
                logger.info(
                    f"Summarize reasoning preview ({len(reasoning_text)} chars): {preview}"
                )

            self._prev_resp_id = None
            self._latest_tool_call_id = "call_placeholder"
            logger.info("Performed LLM summarization and reset conversation thread")
        except Exception as e:
            logger.warning("Summarization failed.")
            raise e

    def _compute_environment_change(
        self,
        prev_frame: Optional[list[list[list[int]]]],
        curr_frame: list[list[list[int]]],
    ) -> tuple[bool, int]:
        """Compare previous and current 3D grids; return (changed?, changed_cells).

        If no previous frame, returns (False, 0).
        """
        if not prev_frame:
            return False, 0
        try:
            changed = 0
            for layer_idx in range(min(len(prev_frame), len(curr_frame))):
                prev_layer = prev_frame[layer_idx]
                curr_layer = curr_frame[layer_idx]
                height = min(len(prev_layer), len(curr_layer))
                for y in range(height):
                    prev_row = prev_layer[y]
                    curr_row = curr_layer[y]
                    width = min(len(prev_row), len(curr_row))
                    for x in range(width):
                        if prev_row[x] != curr_row[x]:
                            changed += 1
            return (changed > 0), changed
        except Exception:
            # Be robust to malformed frames
            return False, 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        tools = self.build_tools()

        # on the very first turn force RESET to start the game
        if self.action_counter == 0 and latest_frame.state is GameState.NOT_PLAYED:
            action = GameAction.RESET
            return action

        current_grid = latest_frame.frame[-1] if latest_frame.frame else []
        grid_base64 = self.generate_grid_image_with_zone(
            current_grid, cell_size=self.CELL_SIZE
        )

        # Detect environment change w.r.t the previous frame
        prev_frame_obj = frames[-2] if len(frames) >= 2 else None
        prev_grid = prev_frame_obj.frame if prev_frame_obj else None
        env_changed, changed_cells = self._compute_environment_change(
            prev_grid, latest_frame.frame
        )

        # Last completed action name from the latest frame
        last_action_name = (
            latest_frame.action_input.id.name if latest_frame.action_input else None
        )

        function_response = build_function_call_output_text(
            state=latest_frame.state.name,
            score=latest_frame.score,
            action_name=last_action_name,
            env_changed=env_changed,
            changed_cells=changed_cells,
            # grid=self.pretty_print_3d(latest_frame.frame),
        )

        # Summarize and reset if token usage is very high
        if self.current_thread_tokens >= self.TOKEN_SUMMARIZE_THRESHOLD:
            self._summarize(
                latest_frame,
                (self._latest_tool_call_id, function_response, grid_base64),
            )

        # Defaults in case the model doesn't pick a tool
        name = GameAction.ACTION5.name
        arguments: Optional[str] = None

        logger.info("Querying Agent for action...")
        try:
            if self.action_counter == 1:
                kickoff_text = "Begin the game. Use the attached grid image and environment snapshot to choose exactly one action."
                logger.info(f"Kickoff text:\n{kickoff_text}\n\n{function_response}")
                response = self._responses_create(
                    developer_text=get_developer_prompt(),
                    user_text=f"{kickoff_text}\n\n{function_response}",
                    image_base64=grid_base64,
                    tools=tools,
                    tool_choice="required",
                )
            else:
                if self._prev_resp_id is None:
                    resume_text = "Resume from handoff. Read the <handoff_summary> above, verify against the attached grid image, then call exactly one action tool from {RESET, ACTION1..ACTION6}. If ACTION6, include integer x y in [0,63]."
                    logger.info(f"Resume text:\n{resume_text}\n\n{function_response}")
                    response = self._responses_create(
                        developer_text=get_developer_prompt(),
                        user_text=f"{self.running_summary}\n\n{resume_text}",
                        image_base64=grid_base64,
                        tool_choice="required",
                        tools=tools,
                    )
                else:
                    logger.info(f"Function call output:\n{function_response}")
                    response = self._responses_create(
                        developer_text=get_developer_prompt(),
                        function_call_output=(
                            self._latest_tool_call_id,
                            function_response,
                            grid_base64,
                        )
                        if self._latest_tool_call_id
                        and self._latest_tool_call_id != "call_placeholder"
                        else None,
                        tools=tools,
                        tool_choice="required",
                        previous_response_id=self._prev_resp_id,
                    )
            self._prev_resp_id = response.id
        except openai.BadRequestError as e:
            if "context_length_exceeded" in str(e).lower():
                # TODO: broken, this won't work with gpt-5, use 4.1 instead
                logger.warning(
                    "Action context too long; compacting and retrying once..."
                )
                self._summarize(
                    latest_frame,
                    (self._latest_tool_call_id, function_response, grid_base64),
                )
            else:
                raise e

        # Observation-and-action loop: allow the model to jot notes via OBSERVE before choosing an action tool
        observations_this_turn: list[str] = []
        final_tool_name: Optional[str] = None
        final_arguments: Optional[str] = None

        while True:
            preamble_text = ""
            reasoning_text = ""
            for item in response.output or []:
                if item.type == "message":
                    for elem in item.content or []:
                        if elem.type == "output_text" and elem.text:
                            preamble_text += elem.text or ""
                elif item.type == "reasoning":
                    for summary in item.summary:
                        reasoning_text += summary.text or ""

            self.track_tokens(response.usage, preamble_text, reasoning_text)

            if reasoning_text:
                preview = (
                    reasoning_text
                    if len(reasoning_text) <= self.REASONING_LOG_PREVIEW_CHARS
                    else reasoning_text[: self.REASONING_LOG_PREVIEW_CHARS] + "..."
                )
                logger.info(
                    f"Reasoning summary preview ({len(reasoning_text)} chars): {preview}"
                )

            # Look for a tool call in the response
            tool_call_item = None
            for item in response.output or []:
                if item.type == "function_call":
                    tool_call_item = item
                    break

            if tool_call_item is None:
                logger.debug("No function_call found; defaulting to ACTION5")
                break

            self._latest_tool_call_id = tool_call_item.call_id
            tool_name = tool_call_item.name
            tool_args = tool_call_item.arguments
            logger.debug(
                f"Assistant: {tool_name} ({self._latest_tool_call_id}) {tool_args}"
            )

            if tool_name == "OBSERVE":
                note_text = ""
                try:
                    data = json.loads(tool_args) or {}
                    note_text = str(data.get("note", "")).strip()
                except Exception as e:
                    logger.warning(f"OBSERVE arguments JSON parse error: {e}")
                if note_text:
                    self.observations.append(note_text)
                    observations_this_turn.append(note_text)
                    if hasattr(self, "recorder") and not self.is_playback:
                        self.recorder.record({"observation": note_text})
                ack_text = f"OBSERVE received ({len(observations_this_turn)} this turn). Continue; choose a game action when ready."
                response = self._responses_create(
                    developer_text=get_developer_prompt(),
                    function_call_output=(
                        self._latest_tool_call_id,
                        ack_text,
                        grid_base64,
                    ),
                    previous_response_id=self._prev_resp_id,
                    tools=tools,
                    tool_choice="required",
                )
                self._prev_resp_id = response.id
                continue

            # If it's a known game action, use it as the final selection
            try:
                _ = GameAction.from_name(tool_name)
                final_tool_name = tool_name
                final_arguments = tool_args
                break
            except Exception:
                # Unknown tool call; ask the model to try again
                logger.warning(
                    f"Unknown tool call: {tool_name}; requesting valid action."
                )
                ack_text = "Unknown tool. Use OBSERVE to jot notes or choose one action tool from {RESET, ACTION1..ACTION6}."
                response = self._responses_create(
                    developer_text=get_developer_prompt(),
                    function_call_output=(
                        self._latest_tool_call_id,
                        ack_text,
                        grid_base64,
                    ),
                    previous_response_id=self._prev_resp_id,
                    tools=tools,
                    tool_choice="required",
                )
                self._prev_resp_id = response.id
                continue

        # Finalize action selection
        if final_tool_name is not None:
            name = final_tool_name
            arguments = final_arguments
        else:
            logger.debug("Falling back to default action ACTION5")

        # We no longer record assistant messages locally
        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        action = GameAction.from_name(action_id)
        action.set_data(data)
        reasoning_payload: dict[str, Any] = (
            action.reasoning if isinstance(action.reasoning, dict) else {}
        )
        reasoning_payload.update(
            {
                "observations": list(self.observations),
                "observations_turn": list(observations_this_turn),
            }
        )
        action.reasoning = reasoning_payload
        self._last_observations_turn = list(observations_this_turn)

        return action

    def track_tokens(
        self, usage: ResponseUsage, message: str = "", reasoning_text: str = ""
    ) -> None:
        total_tokens = usage.total_tokens
        input_tokens = usage.input_tokens
        cached_tokens = usage.input_tokens_details.cached_tokens
        output_tokens = usage.output_tokens

        self.token_counter += total_tokens
        self.current_thread_tokens = total_tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "input_tokens": input_tokens,
                    "cached_tokens": cached_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "message": message,
                    "reasoning": reasoning_text,
                }
            )
        logger.info(
            f"Tokens: in={input_tokens} cached={cached_tokens} out={output_tokens} total={total_tokens}; running_total={self.token_counter}"
        )

    def build_tools(self) -> list[dict[str, Any]]:
        """Build Responses-native tool descriptors for game actions."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        observe_params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Brief internal observation or note about the current grid/state.",
                }
            },
            "required": ["note"],
            "additionalProperties": False,
        }
        action6_params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "string",
                    "description": "Coordinate X which must be Int<0,63>",
                },
                "y": {
                    "type": "string",
                    "description": "Coordinate Y which must be Int<0,63>",
                },
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        return [
            {
                "type": "function",
                "name": "OBSERVE",
                "description": "Write a short internal note to help you think; can be called multiple times before choosing an action.",
                "parameters": observe_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
                "strict": True,
            },
            {
                "type": "function",
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": action6_params,
                "strict": True,
            },
        ]

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_tools": self.build_tools(),
                    "llm_running_summary": self.running_summary,
                    "last_state": {
                        "state": self.frames[-1].state.name,
                        "score": self.frames[-1].score,
                        "frame_preview": self.pretty_print_3d(self.frames[-1].frame),
                    },
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 200
    MODEL = "gpt-5"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._total_reasoning_tokens = 0
        self._last_message: str = ""
        self._last_reasoning_text: str = ""

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "message": self._last_message,
            "reasoning_preview": self._last_reasoning_text[:200] + "..."
            if len(self._last_reasoning_text) > 200
            else self._last_reasoning_text,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
        }

        return action

    def track_tokens(
        self, usage: ResponseUsage, message: str = "", reasoning_text: str = ""
    ) -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(usage, message, reasoning_text)

        # Persist latest message and reasoning text
        self._last_message = message
        self._last_reasoning_text = reasoning_text
        self._last_reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        self._total_reasoning_tokens += self._last_reasoning_tokens
