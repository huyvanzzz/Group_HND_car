"""Validate the workflow's dependency closure in a shared notebook environment."""
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def validate_dependencies(requirements, *, distribution=metadata.distribution):
    """Reject missing/incompatible active dependencies, including requested extras.

    Unlike global ``pip check``, unrelated preinstalled applications are not roots.
    No model or CUDA imports are needed for this metadata-only check.
    """
    pending = [Requirement(value) for value in requirements]
    visited = set()
    versions = {}
    errors = set()
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        try:
            package = distribution(name)
        except metadata.PackageNotFoundError:
            errors.add(f'{requirement}: not installed')
            continue
        versions[name] = package.version
        if requirement.specifier and not requirement.specifier.contains(package.version, prereleases=True):
            errors.add(f'{requirement}: installed {package.version}')
        extras = frozenset(requirement.extras)
        key = (name, extras)
        if key in visited:
            continue
        visited.add(key)
        for value in package.requires or ():
            child = Requirement(value)
            if child.marker is None or any(child.marker.evaluate({'extra': extra})
                                           for extra in ('', *extras)):
                pending.append(child)
    if errors:
        raise RuntimeError('SmolVLM dependency validation failed:\n' + '\n'.join(sorted(errors)))
    return dict(sorted(versions.items()))
