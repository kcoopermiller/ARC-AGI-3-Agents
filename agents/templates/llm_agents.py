import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from openai import OpenAI as OpenAIClient
from openai.types.responses import ResponseInputParam

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 10
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = "high"
    MESSAGE_LIMIT: int = 10
    MODEL: str = "gpt-5"
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        self._last_action_resp_id: Optional[str] = None
        self._prev_resp_id: Optional[str] = None

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        name += f".{self.REASONING_EFFORT}"
        return name

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

        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        tools = self.build_tools()

        # on the very first turn force RESET to start the game
        if self.action_counter == 0 and latest_frame.state is GameState.NOT_PLAYED:
            return GameAction.RESET

        # let the agent comment observations before choosing action
        # on the first turn, this will be in response to RESET action
        function_response = self.build_func_resp_prompt(latest_frame)

        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            try:
                # Prefer sending the previous tool result back as function_call_output tied to the last action's tool_call id
                if (
                    self._latest_tool_call_id
                    and self._latest_tool_call_id != "call_12345"
                    and self._last_action_resp_id
                ):
                    observation_input: ResponseInputParam = [
                        {
                            "type": "function_call_output",
                            "call_id": self._latest_tool_call_id,
                            "output": str(function_response),
                        }
                    ]
                    create_kwargs: dict[str, Any] = {
                        "model": self.MODEL,
                        "input": observation_input,
                        "previous_response_id": self._last_action_resp_id,
                    }
                else:
                    # Fallback: simple observation text
                    observation_input = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": str(function_response)}
                            ],
                        }
                    ]
                    create_kwargs = {
                        "model": self.MODEL,
                        "input": observation_input,
                    }
                create_kwargs["reasoning"] = {"effort": self.REASONING_EFFORT}
                response = client.responses.create(**create_kwargs)
                self._prev_resp_id = getattr(response, "id", None)
            except openai.BadRequestError as e:
                try:
                    logger.info(
                        "Observation request: %s",
                        {
                            "model": self.MODEL,
                            "input": observation_input,
                            "previous_response_id": self._last_action_resp_id,
                            "reasoning": {"effort": self.REASONING_EFFORT},
                        },
                    )
                finally:
                    raise e
            # Extract assistant text from Responses API
            assistant_text = ""
            try:
                if getattr(response, "output_text", None):
                    assistant_text = response.output_text or ""
                else:
                    for item in getattr(response, "output", []) or []:
                        if getattr(item, "type", None) == "message":
                            for elem in getattr(item, "content", []) or []:
                                if getattr(
                                    elem, "type", None
                                ) == "output_text" and getattr(elem, "text", None):
                                    assistant_text += elem.text or ""
            except Exception:
                assistant_text = ""

            self.track_tokens(response.usage.total_tokens, assistant_text)
            logger.info(f"Assistant: {assistant_text}")

        # now ask for the next action
        user_prompt = self.build_user_prompt(latest_frame)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None

        # Use Responses API with tools for action selection
        logger.info("Sending to Assistant for action...")
        try:
            # Build a minimal Responses input for action selection
            action_input: ResponseInputParam = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(user_prompt)}],
                }
            ]
            # If we have just had a tool call ID, include the function_call_output first
            if self._latest_tool_call_id and self._latest_tool_call_id != "call_12345":
                action_input.insert(
                    0,
                    {
                        "type": "function_call_output",
                        "call_id": self._latest_tool_call_id,
                        "output": str(function_response),
                    },
                )
            create_kwargs = {
                "model": self.MODEL,
                "input": action_input,
                "tools": tools,
                "tool_choice": "required",
            }
            create_kwargs["reasoning"] = {"effort": self.REASONING_EFFORT}
            if self._prev_resp_id:
                create_kwargs["previous_response_id"] = self._prev_resp_id
            response = client.responses.create(**create_kwargs)
            self._prev_resp_id = getattr(response, "id", None)
            self._last_action_resp_id = self._prev_resp_id
        except openai.BadRequestError as e:
            try:
                logger.info(
                    "Action request: %s",
                    {
                        "model": self.MODEL,
                        "input": action_input,
                        "tools": [t.get("name") for t in tools],
                        "tool_choice": "required",
                        "previous_response_id": self._prev_resp_id,
                        "reasoning": {"effort": self.REASONING_EFFORT},
                    },
                )
            finally:
                raise e

        # Track tokens (Responses API)
        try:
            self.track_tokens(response.usage.total_tokens)
        except Exception:
            pass

        # Parse function call from Responses API
        tool_call_item = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                tool_call_item = item
                break

        if tool_call_item is not None:
            self._latest_tool_call_id = getattr(tool_call_item, "call_id", "call_12345")
            name = getattr(tool_call_item, "name", name)
            arguments = getattr(tool_call_item, "arguments", arguments)
            logger.debug(f"Assistant: {name} ({self._latest_tool_call_id}) {arguments}")
        else:
            logger.debug("No function_call found; defaulting to ACTION5")

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
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def build_tools(self) -> list[dict[str, Any]]:
        """Build Responses-native tool descriptors for game actions."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
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

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.score,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

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
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 5
    DO_OBSERVATION = True
    MODEL = "gpt-5"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Capture reasoning tokens from Responses API usage if available."""
        try:
            # Prefer Responses API field
            if (
                hasattr(response, "usage")
                and hasattr(response.usage, "output_tokens_details")
                and hasattr(response.usage.output_tokens_details, "reasoning_tokens")
            ):
                self._last_reasoning_tokens = (
                    response.usage.output_tokens_details.reasoning_tokens or 0
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
        except Exception:
            pass


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLM(LLM, Agent):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "o3"
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o3 models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        try:
            if (
                hasattr(response, "usage")
                and hasattr(response.usage, "output_tokens_details")
                and hasattr(response.usage.output_tokens_details, "reasoning_tokens")
            ):
                self._last_reasoning_tokens = (
                    response.usage.output_tokens_details.reasoning_tokens or 0
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
        except Exception:
            pass

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

You are playing a game called LockSmith. Rules and strategy:
* RESET: start over, ACTION1: move up, ACTION2: move down, ACTION3: move left, ACTION4: move right (ACTION5 and ACTION6 do nothing in this game)
* you may may one action per turn
* your goal is find and collect a matching key then touch the exit door
* 6 levels total, score shows which level, complete all levels to win (grid row 62)
* start each level with limited energy. you GAME_OVER if you run out (grid row 61)
* the player is a 4x4 square: [[X,X,X,X],[0,0,0,X],[4,4,4,X],[4,4,4,X]] where X is transparent to the background
* the grid represents a birds-eye view of the level
* walls are made of INT<10>, you cannot move through a wall
* walkable floor area is INT<8>
* you can refill energy by touching energy pills (a 2x2 of INT<6>)
* current key is shown in bottom-left of entire grid
* the exit door is a 4x4 square with INT<11> border
* to find a new key shape, touch the key rotator, a 4x4 square denoted by INT<9> and INT<4> in the top-left corner of the square
* to find a new key color, touch the color rotator, a 4x4 square denoted by INT<9> and INT<2> and in the bottom-left corner of the square
* to rotate more than once, move 1 space away from the rotator and back on
* continue rotating the shape and color of the key until the key matches the one inside the exit door (scaled down 2X)
* if the grid does not change after an action, you probably tried to move into a wall

An example of a good strategy observation:
The player 4x4 made of INT<4> and INT<0> is standing below a wall of INT<10>, so I cannot move up anymore and should
move left towards the rotator with INT<11>.

# TURN:
Call exactly one action.
        """.format()
        )


# Example of a custom LLM agent
class MyCustomLLM(LLM):
    """Template for creating your own custom LLM agent."""

    MAX_ACTIONS = 80
    MODEL = "gpt-4o-mini"
    DO_OBSERVATION = True

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# CUSTOM INSTRUCTIONS:
Add your game instructions and strategy here.
For example, explain the game rules, objectives, and optimal strategies.

# TURN:
Call exactly one action.
        """.format()
        )
