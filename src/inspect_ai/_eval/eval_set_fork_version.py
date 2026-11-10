"""Version numbers for the METR fork's own eval-set protocol additions.

Upstream numbers its selection and capture schemas 1, 2, 3, ... A fork that
took the next number for an addition of its own would collide with whatever
upstream numbers next, and a runner that checks the number could then accept
an upstream inspect that lacks the addition. So a fork addition is numbered

    9000 + 10 * (upstream's version) + (its position in the fork's patch stack)

which upstream never reaches, and which changes whenever the fork is rebased
onto a new upstream version. A patch dropped without being merged upstream
leaves its position unused, so no number ever means two things.
"""

_FORK_RANGE = 9000


def fork_protocol_version(upstream: int, position: int) -> int:
    """Version of the fork's `position`-th addition on top of upstream's `upstream`.

    Args:
        upstream: The schema version upstream defines.
        position: Position of the addition in the fork's patch stack, from 1.

    Returns:
        The fork's version number.

    Raises:
        ValueError: If `position` is not 1 to 9 (a tenth would carry into upstream's digits) or `upstream` is not 1 to 99.
    """
    if not 1 <= position <= 9 or not 1 <= upstream <= 99:
        raise ValueError(
            f"No fork version for position {position} on upstream version {upstream}."
        )
    return _FORK_RANGE + 10 * upstream + position


def protocol_version_understood(version: int, upstream: int, current: int) -> bool:
    """Whether a build can read a document that declares `version`.

    Args:
        version: The version the document declares.
        upstream: The newest upstream version the build is based on.
        current: The build's own version: `upstream`, or a fork version on it.

    Returns:
        `True` for upstream's versions up to `upstream` and for the fork's versions up to `current`. A version between the two is a later upstream one, which a fork version being numerically larger must not let through.
    """
    return version <= upstream or _FORK_RANGE < version <= current
