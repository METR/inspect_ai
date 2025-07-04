import abc
import random
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, Field, ValidationError
from typing_extensions import override

from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.model import ChatMessage
from inspect_ai.util import SandboxEnvironmentSpec, SandboxEnvironmentType
from inspect_ai.util._sandbox.environment import resolve_sandbox_environment

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

MT = TypeVar("MT", bound=BaseModel)


class Sample(BaseModel):
    r"""Sample for an evaluation task."""

    def __init__(
        self,
        input: str | list[ChatMessage],
        choices: list[str] | None = None,
        target: str | list[str] = "",
        id: int | str | None = None,
        metadata: dict[str, Any] | None = None,
        sandbox: SandboxEnvironmentType | None = None,
        files: dict[str, str] | None = None,
        setup: str | None = None,
    ) -> None:
        r"""Create a Sample.

        Args:
            input: The input to be submitted to the model.
            choices: Optional. List of available answer choices
                (used only for multiple-choice evals).
            target: Optional. Ideal target output. May be a literal value
                or narrative text to be used by a model grader.
            id: Optional. Unique identifier for sample.
            metadata: Optional. Arbitrary metadata associated with the sample.
            sandbox: Optional. Sandbox specification for this sample.
            files: Optional. Files that go along with the sample (copied to
                SandboxEnvironment). Files can be paths, inline text, or inline binary (base64 encoded data URL).
            setup: Optional. Setup script to run for sample (run
                within default SandboxEnvironment).
        """
        super().__init__(
            input=input,
            choices=choices,
            target=target,
            id=id,
            metadata=metadata,
            sandbox=resolve_sandbox_environment(sandbox),
            files=files,
            setup=setup,
        )

    input: str | list[ChatMessage]
    """The input to be submitted to the model."""

    choices: list[str] | None = Field(default=None)
    """List of available answer choices (used only for multiple-choice evals)."""

    target: str | list[str] = Field(default_factory=str)
    """Ideal target output. May be a literal value or narrative text to be used by a model grader."""

    id: int | str | None = Field(default=None)
    """Unique identifier for sample."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Arbitrary metadata associated with the sample."""

    def metadata_as(self, metadata_cls: Type[MT]) -> MT:
        """Metadata as a Pydantic model.

        Args:
           metadata_cls: BaseModel derived class.

        Returns:
           BaseModel: Instance of metadata_cls.
        """
        if self.metadata is None:
            raise ValueError("Sample does not have metadata")

        return metadata_as(self.metadata, metadata_cls)

    sandbox: SandboxEnvironmentSpec | None = Field(default=None)
    """Sandbox environment type and optional config file."""

    files: dict[str, str] | None = Field(default=None)
    """Files that go along with the sample (copied to SandboxEnvironment)"""

    setup: str | None = Field(default=None)
    """Setup script to run for sample (run within default SandboxEnvironment)."""


def sample_input_len(sample: Sample) -> int:
    """Measures the length of a samples `input` field.

    The default length function use in `Dataset.sort()`.

    Args:
        sample (Sample): A Sample to be used in an evaluation task.
    """
    return (
        len(sample.input)
        if isinstance(sample.input, str)
        else sum(len(inp.text) for inp in sample.input)
    )


DatasetRecord = dict[str, Any]

DatasetReader = Iterator[DatasetRecord]


class Dataset(Sequence[Sample], abc.ABC):
    r"""A sequence of Sample objects.

    Datasets provide sequential access (via conventional indexes or slicing)
    to a collection of Sample objects.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str | None: ...

    @property
    @abc.abstractmethod
    def location(self) -> str | None: ...

    @property
    @abc.abstractmethod
    def shuffled(self) -> bool: ...

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> "Dataset": ...

    @abc.abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, "Dataset"]: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def sort(
        self,
        reverse: bool = False,
        key: Callable[[Sample], "SupportsRichComparison"] = sample_input_len,
    ) -> None:
        """Sort the dataset (in place) in ascending order and return None.

        If a key function is given, apply it once to each list item and sort them, ascending or descending, according to their function values.

        The key function defaults to measuring the length of the sample's input field.

        Args:
            reverse: If `Treu`, sort in descending order. Defaults to False.
            key: a callable mapping each item to a numeric value (optional, defaults to sample_input_len).
        """

    @abc.abstractmethod
    def filter(
        self, predicate: Callable[[Sample], bool], name: str | None = None
    ) -> "Dataset":
        """Filter the dataset using a predicate.

        Args:
          predicate: Filtering function.
          name: Name for filtered dataset (optional).

        Returns:
          Filtered dataset.
        """

    @abc.abstractmethod
    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the order of the dataset (in place).

        Args:
           seed: Random seed for shuffling (optional).
        """

    @abc.abstractmethod
    def shuffle_choices(self, seed: int | None = None) -> None:
        """Shuffle the order of the choices with each sample.

        Args:
           seed: Random seed for shuffling (optional).
        """


@dataclass
class FieldSpec:
    r"""Specification for mapping data source fields to sample fields."""

    input: str = field(default="input")
    """Name of the field containing the sample input."""

    target: str = field(default="target")
    """Name of the field containing the sample target."""

    choices: str = field(default="choices")
    """Name of field containing the list of answer choices."""

    id: str = field(default="id")
    """ Unique identifier for the sample."""

    metadata: list[str] | Type[BaseModel] | None = field(default=None)
    """List of additional field names that should be read as metadata."""

    sandbox: str = field(default="sandbox")
    """Sandbox type along with optional config file."""

    files: str = field(default="files")
    """Files that go along wtih the sample."""

    setup: str = field(default="setup")
    """Setup script to run for sample (run within default SandboxEnvironment)."""


RecordToSample = Callable[[DatasetRecord], Sample | list[Sample]]
r"""Callable that maps raw dictionary record to a Sample."""


class MemoryDataset(Dataset):
    r"""A Dataset stored in memory."""

    def __init__(
        self,
        samples: list[Sample],
        name: str | None = None,
        location: str | None = None,
        shuffled: bool = False,
    ) -> None:
        r"""A dataset of samples held in an in-memory list.

        Datasets provide sequential access (via conventional indexes or slicing)
        to a collection of Sample objects. The ListDataset is explicitly
        initialized with a list that is held in memory.

        Args:
            samples (list[Sample]): The list of sample objects.
            name (str | None): Optional name for dataset.
            location (str | None): Optional location for dataset.
            shuffled (bool): Was the dataset shuffled after reading.
        """
        self.samples = samples
        self._name = name
        self._location = location
        self._shuffled = shuffled

    @override
    @property
    def name(self) -> str | None:
        """Dataset name."""
        return self._name

    @override
    @property
    def location(self) -> str | None:
        """Dataset location."""
        return self._location

    @override
    @property
    def shuffled(self) -> bool:
        """Was the dataset shuffled."""
        return self._shuffled

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Dataset: ...

    @override
    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, Dataset]:
        if isinstance(index, int):
            return self.samples[index]
        else:
            return MemoryDataset(
                samples=self.samples[index],
                name=self.name,
                location=self.location,
                shuffled=self.shuffled,
            )

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def shuffle(self, seed: int | None = None) -> None:
        if seed is not None:
            random.Random(seed).shuffle(self.samples)
        else:
            random.shuffle(self.samples)
        self._shuffled = True

    @override
    def shuffle_choices(self, seed: int | None = None) -> None:
        rand = random.Random(seed)
        for sample in self.samples:
            if not sample.choices:
                continue
            # The original positions
            positions = list(range(len(sample.choices)))

            # Shuffle the choices
            rand.shuffle(positions)
            shuffled_choices = [sample.choices[i] for i in positions]

            # Map of original position / target letter
            position_map = {
                i: answer_character(new_i) for new_i, i in enumerate(positions)
            }

            # Update to the shuffled choices and target
            sample.choices = shuffled_choices
            sample.target = self._remap_target(sample.target, position_map=position_map)

    def _remap_target(
        self, target: str | list[str], position_map: dict[int, str]
    ) -> str | list[str]:
        if isinstance(target, list):
            return [position_map[answer_index(t)] for t in target]
        else:
            return position_map[answer_index(target)]

    @override
    def sort(
        self,
        reverse: bool = False,
        key: Callable[[Sample], "SupportsRichComparison"] = sample_input_len,
    ) -> None:
        self.samples.sort(reverse=reverse, key=key)

    @override
    def filter(
        self, predicate: Callable[[Sample], bool], name: str | None = None
    ) -> "MemoryDataset":
        return MemoryDataset(
            name=name or self.name,
            location=self.location,
            samples=[sample for sample in self if predicate(sample)],
            shuffled=self.shuffled,
        )


def metadata_as(metadata: dict[str, Any], metadata_cls: Type[MT]) -> MT:
    # validate that metadata_cls is frozen
    if not metadata_cls.model_config.get("frozen", False):
        raise ValueError(
            f"Metadata model {metadata_cls.__name__} must have frozen=True"
        )

    # filter to only fields in the model
    model_fields = {
        k: v
        for k, v in metadata.items()
        if k in metadata_cls.__pydantic_fields__.keys()
    }

    # parse and return model instance
    try:
        return metadata_cls(**model_fields)
    except ValidationError as ex:
        raise ValueError(f"Could not parse metadata into {metadata_cls.__name__}: {ex}")
