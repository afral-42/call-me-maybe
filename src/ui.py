import json
import os
import time
from typing import Any, Generator, Self

import numpy as np
import streamlit as st
from htbuilder import div, styles
from htbuilder.units import rem

from llm_sdk import Small_LLM_Model
from src.machine import StateMachine
from src.parsing import build_linked_state_machines
from src.prompts import Prompt
from src.states import StateException
from src.trie import Trie, get_token_trie


class InferenceException(Exception):
    def __init__(self, detail: str) -> None:
        """

        Initialize the inference exception.

        Args:

            detail (str): The detail message for the error.

        """
        super().__init__(f"Inference error: {detail}")


class StreamInferenceEngine:
    """

    Streaming inference engine for constrained generation.

    """

    def __init__(
        self,
        llm: Small_LLM_Model,
        trie: Trie,
        output_file_path: str,
        linked_state_machines: dict[Prompt, StateMachine]
    ) -> None:
        """

        Initialize the stream inference engine.

        Args:

            llm (Small_LLM_Model): The language model.

            trie (Trie): The token trie.

            output_file_path (str): Path to output file.

            linked_state_machines (dict[Prompt, StateMachine]):
                Linked state machines.

        """
        self.machines = linked_state_machines
        self.trie = trie
        self.llm = llm
        self.output_file_path = output_file_path
        self.built_prompts: dict[Prompt, list[int]] = {}
        self.results: dict[Prompt, str] = {
            p: "" for p in linked_state_machines
        }
        self._batch_encode()

    def _batch_encode(self) -> None:
        """

        Encode all prompts into token IDs.

        """
        for prompt in self.machines.keys():
            self.built_prompts[prompt] = (
                self.llm.encode(str(prompt))[0].tolist()
            )

    def stream_run(
        self, prompt: Prompt
    ) -> Generator[dict[str, Any], None, None]:
        """

        Run streaming inference for a prompt.

        Args:

            prompt (Prompt): The prompt to process.

        Yields:

            dict[str, Any]: Step information.

        """
        ids = self.built_prompts[prompt]
        machine = self.machines[prompt]

        try:
            while not machine.is_done():
                authorized = machine.get_authorized_tokens(self.trie)
                static_string = machine.get_static_string()

                if machine.states:
                    state_name = machine.states[0].__class__.__name__
                else:
                    state_name = "State"

                if static_string is not None:
                    tokens = self.llm.encode(static_string)[0].tolist()
                    for token in tokens:
                        self.built_prompts[prompt].append(token)
                    self.results[prompt] += static_string
                    machine.consume(static_string)
                    yield {
                        "token": static_string,
                        "state": state_name,
                        "allowed": [static_string]
                    }
                    continue

                if len(authorized) == 1:
                    best_token = list(authorized)[0]
                else:
                    logits = self.llm.get_logits_from_input_ids(ids)
                    best_token = self._get_best_token(authorized, logits)

                best_token_str = self.llm.decode([best_token])
                cleaned_tokens = machine.intercept_token(
                    best_token_str, best_token, self.llm
                )

                allowed_strs = [
                    self.llm.decode([t]) for t in list(authorized)[:8]
                ]

                for token in cleaned_tokens:
                    token_str = self.llm.decode([token])
                    self.results[prompt] += token_str
                    self.built_prompts[prompt].append(token)
                    machine.consume(token_str)
                    yield {
                        "token": token_str,
                        "state": state_name,
                        "allowed": allowed_strs
                    }
        except StateException as e:
            yield {"error": str(e)}

    def _get_best_token(
        self, authorized: set[int], logits: list[float]
    ) -> int:
        """

        Get the best token from authorized tokens.

        Args:

            authorized (set[int]): Authorized token IDs.

            logits (list[float]): Logits.

        Returns:

            int: Best token ID.

        """
        if len(authorized) > 1000:
            auth_list = list(authorized)
            auth_logits = [logits[i] for i in auth_list]
            best_index = int(np.argmax(auth_logits))
            return auth_list[best_index]
        return max(authorized, key=lambda i: logits[i])

    @classmethod
    def build_engine(
        cls,
        input_path: str,
        functions_definition_path: str,
        output_path: str
    ) -> Self:
        """

        Build a stream inference engine.

        Args:

            input_path (str): Path to input prompts.

            functions_definition_path (str): Path to functions.

            output_path (str): Path to output.

        Returns:

            StreamInferenceEngine: The engine.

        """
        llm = Small_LLM_Model()
        trie = get_token_trie(llm)
        machines = build_linked_state_machines(
            functions_definition_path, input_path
        )
        return cls(
            llm=llm,
            trie=trie,
            output_file_path=output_path,
            linked_state_machines=machines
        )


st.set_page_config(
    page_title="Call me Maybe",
    page_icon="📞",
    layout="wide"
)

st.html(div(style=styles(font_size=rem(4), line_height=1))["❉"])
title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Call me Maybe", anchor=False)

os.makedirs("data", exist_ok=True)

with st.sidebar:
    st.subheader("Configuration")
    st.caption(
        "This project has been created "
        "as part of the 42 curriculum by abounoua"
    )

    uploaded_file = st.file_uploader(
        "Upload functions definitions (JSON)", type=["json"]
    )

    if uploaded_file is not None:
        selected_def = "data/temp_ui_defs.json"
        with open(selected_def, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Functions loaded!")
    else:
        selected_def = "data/input/functions_definition.json"
        st.info("Using default definitions.")

    st.divider()


@st.cache_resource
def load_llm_trie() -> tuple[Small_LLM_Model, Trie]:
    """

    Load and cache the LLM and trie.

    Returns:

        tuple[Small_LLM_Model, Trie]: The LLM and trie.

    """
    model = Small_LLM_Model()
    model_trie = get_token_trie(model)
    return model, model_trie


try:
    llm, trie = load_llm_trie()
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="json")
        else:
            st.markdown(msg["content"])

if user_message := st.chat_input("Ask a follow-up..."):
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.text(user_message)

    temp_input_path = "data/temp_ui_input.json"
    with open(temp_input_path, "w") as input:
        json.dump([{"prompt": user_message}], input)

    with st.chat_message("assistant"):
        with st.spinner("Compiling constraints..."):
            try:
                machines = build_linked_state_machines(
                    selected_def, temp_input_path
                )
                engine = StreamInferenceEngine(
                    llm=llm,
                    trie=trie,
                    output_file_path="data/output/temp_output.json",
                    linked_state_machines=machines
                )
            except Exception as e:
                st.error(f"Failed to build linked states: {e}")
                st.stop()

        prompt_obj = list(engine.machines.keys())[0]

        response_placeholder = st.empty()
        st.write("---")

        state_placeholder = st.empty()
        st.caption("🛡️ **Active Constraints**")
        token_list_placeholder = st.empty()

        full_response = ""

        for step in engine.stream_run(prompt_obj):
            if "error" in step:
                st.error(f"FSM Error: {step['error']}")
                break

            full_response += step["token"]
            response_placeholder.code(full_response, language="json")

            with token_list_placeholder.container():
                st.pills(
                    label=f"Allowed by {step['state']}:",
                    options=step["allowed"],
                    selection_mode="single",
                    disabled=True,
                    key=f"pills_{len(full_response)}"
                )
            time.sleep(0.10)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
