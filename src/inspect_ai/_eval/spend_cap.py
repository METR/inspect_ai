"""Eval-set cost accounting and cancellation across attempts."""

import logging
import math
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import anyio
from pydantic import BaseModel, Field

from inspect_ai._util.asyncfiles import AsyncFilesystem
from inspect_ai._util.error import PrerequisiteError
from inspect_ai._util.file import absolute_file_path, filesystem, local_path

logger = logging.getLogger(__name__)

SPEND_FILE = ".eval-set-spend.json"
FLUSH_INTERVAL = 30
SPEND_CAP_REACHED = "Eval-set spend cap reached"


class SpendLedger(BaseModel):
    spent_usd: float = Field(ge=0, allow_inf_nan=False)


def validate_spend_cap(limit: float) -> None:
    if not math.isfinite(limit) or limit <= 0:
        raise PrerequisiteError(
            f"spend_cap must be a finite, positive dollar amount: {limit}"
        )


class SpendCap:
    """Track new charges independently of reused sample results.

    The eval-set owns this object across its retry runs. Each run owns one
    async writer and registers its task cancellation handles. All updates
    happen on the eval's event loop, so counter updates need no lock.
    """

    def __init__(self, limit: float, log_dir: str) -> None:
        self.limit = limit
        self.spent = 0.0
        self.tripped = False
        self._path = f"{absolute_file_path(log_dir).rstrip('/')}/{SPEND_FILE}"
        self._loaded = False
        self._saved = 0.0
        self._tasks: set[Callable[[], None]] = set()
        self._unpriced: set[str] = set()
        self._warned = False

    def record(self, model: str, cost: float | None) -> None:
        if cost is None:
            if model not in self._unpriced:
                self._unpriced.add(model)
                logger.warning(
                    "The spend cap cannot count %s: no cost is available. "
                    "Supply prices with model_cost_config.",
                    model,
                )
            return
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid model cost for {model}: {cost}")
        self.spent += cost
        if self.tripped:
            return
        if self.spent >= self.limit:
            self.tripped = True
            logger.warning(
                "%s: $%.2f of $%.2f. "
                "Cancelling unfinished work without scoring. "
                "Resume with a higher spend_cap to continue.",
                SPEND_CAP_REACHED,
                self.spent,
                self.limit,
            )
            for cancel in tuple(self._tasks):
                cancel()
        elif not self._warned and self.spent >= self.limit * 0.8:
            self._warned = True
            logger.warning(
                "Eval set has spent $%.2f of its $%.2f spend cap",
                self.spent,
                self.limit,
            )

    @contextmanager
    def task(self, cancel: Callable[[], None]) -> Iterator[None]:
        self._tasks.add(cancel)
        try:
            if self.tripped:
                cancel()
            yield
        finally:
            self._tasks.discard(cancel)

    async def _load(self, fs: AsyncFilesystem) -> None:
        if self._loaded:
            return
        try:
            data = await fs.read_file(self._path)
        except FileNotFoundError:
            pass
        except Exception as ex:
            raise PrerequisiteError(
                f"Cannot read saved eval-set spend at {self._path}: {ex}"
            ) from ex
        else:
            try:
                self.spent = SpendLedger.model_validate_json(data).spent_usd
            except ValueError as ex:
                raise PrerequisiteError(
                    f"Invalid saved eval-set spend at {self._path}: {ex}"
                ) from ex
        self._saved = self.spent
        self._warned = self.spent >= self.limit * 0.8
        if self.spent >= self.limit:
            raise PrerequisiteError(
                f"Eval set has already spent ${self.spent:.2f} of its "
                f"${self.limit:.2f} spend cap. Resume with a higher spend_cap."
            )
        self._loaded = True

    async def _flush(self, fs: AsyncFilesystem) -> None:
        if self.spent == self._saved:
            return
        snapshot = self.spent
        data = SpendLedger(spent_usd=snapshot).model_dump_json().encode()
        # A local write truncates the file first, so a process killed mid-write
        # would leave a total that blocks the next start. Replace it instead.
        local = filesystem(self._path).is_local()
        target = f"{self._path}.tmp" if local else self._path
        try:
            await fs.write_file(target, data)
            if local:
                os.replace(local_path(target), local_path(self._path))
        except Exception as ex:
            logger.warning("Could not save eval-set spend to %s: %s", self._path, ex)
        else:
            self._saved = snapshot

    async def _write_periodically(self, fs: AsyncFilesystem) -> None:
        while True:
            await anyio.sleep(FLUSH_INTERVAL)
            await self._flush(fs)

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        async with AsyncFilesystem() as fs:
            await self._load(fs)
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._write_periodically, fs)
                    try:
                        yield
                    finally:
                        tg.cancel_scope.cancel()
            finally:
                # The periodic writer has been joined, so an interrupted older
                # write cannot overwrite this one.
                with anyio.CancelScope(shield=True):
                    await self._flush(fs)


_spend_cap: ContextVar[SpendCap | None] = ContextVar("eval_set_spend_cap", default=None)


@contextmanager
def spend_cap_scope(cap: SpendCap | None) -> Iterator[None]:
    token = _spend_cap.set(cap)
    try:
        yield
    finally:
        _spend_cap.reset(token)


def record_spend(model: str, cost: float | None) -> None:
    cap = _spend_cap.get()
    if cap is not None:
        cap.record(model, cost)


def spend_cap_active() -> bool:
    return _spend_cap.get() is not None


def spend_cap_reached() -> bool:
    cap = _spend_cap.get()
    return cap is not None and cap.tripped


@contextmanager
def spend_cap_task(cancel: Callable[[], None]) -> Iterator[None]:
    cap = _spend_cap.get()
    if cap is None:
        yield
    else:
        with cap.task(cancel):
            yield


@asynccontextmanager
async def spend_cap_run() -> AsyncIterator[None]:
    cap = _spend_cap.get()
    if cap is None:
        yield
    else:
        async with cap.run():
            yield
