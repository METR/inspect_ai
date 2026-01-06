import pytest
from test_helpers.utils import simple_task_state

from inspect_ai.scorer import CORRECT, INCORRECT, Target, answer


@pytest.mark.anyio
async def test_letter_success():
    scorer = answer("letter")
    state = simple_task_state(model_output="ANSWER: B")
    result = await scorer(state, Target(["B"]))

    assert result.text == CORRECT


@pytest.mark.anyio
async def test_letter_failure():
    scorer = answer("letter")
    state = simple_task_state(model_output="ANSWER: B")
    result = await scorer(state, Target(["C"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_word_success():
    scorer = answer("word")
    state = simple_task_state(model_output="ANSWER: Yes")
    result = await scorer(state, Target(["Yes"]))

    assert result.text == CORRECT


@pytest.mark.anyio
async def test_word_failure():
    scorer = answer("letter")
    state = simple_task_state(model_output="ANSWER: Yes")
    result = await scorer(state, Target(["No"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_line_success():
    scorer = answer("line")
    state = simple_task_state(model_output="ANSWER:\nThis is a whole new line")
    result = await scorer(state, Target(["This is a whole new line"]))

    assert result.text == CORRECT


@pytest.mark.anyio
async def test_line_failure():
    scorer = answer("line")
    state = simple_task_state(model_output="ANSWER:\nThis is a whole new line")
    result = await scorer(state, Target(["This doesn't match does it?"]))

    assert result.text == INCORRECT


@pytest.mark.anyio
async def test_prefers_last_non_blank_line():
    scorer = answer("line")
    state = simple_task_state(
        model_output=(
            "True answer: only chapter 2 remains ⇒ 1.\n\n"
            "ANSWER: only chapter 2 remains ⇒ 1."
        )
    )

    result = await scorer(state, Target(["only chapter 2 remains ⇒ 1."]))

    assert result.text == CORRECT
    assert result.answer == "only chapter 2 remains ⇒ 1."


@pytest.mark.anyio
async def test_trailing_blank_lines_after_answer():
    scorer = answer("letter")
    state = simple_task_state(model_output="ANSWER: B\n\n\n")

    result = await scorer(state, Target(["B"]))

    assert result.text == CORRECT
