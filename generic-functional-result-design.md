## The Goal

When a user writes:
```python
result = backend.execute(functional, sampling=SamplingReadout(1000), state_tomography=StateTomographyReadout())
```

The type checker should know that:
- `result.readout_results.sampling` is `SamplingReadoutResult` (not `Optional`)
- `result.readout_results.state_tomography` is `StateTomographyReadoutResult` (not `Optional`)
- `result.readout_results.expectation_values` is `None` (accessing `.expected_values` on it is a type error)

## The Key Constraint

Python's type system cannot discriminate elements inside a `list`. If `readout` is `list[ReadoutMethod]`, the type checker can't know which subclasses are in the list. This means **the current `readout: list[ReadoutMethod]` API shape is fundamentally incompatible with strong return-type narrowing**.

To get strong typing, the API must change so the type checker sees each readout method as a separate, named thing. There are two viable shapes: **keyword arguments** and a **typed builder**. Let me walk through both, starting with keywords since it's simpler.

---

### Layer 1: Generic `ReadoutCompositeResults`

The foundation. Three type variables, each constrained to either the concrete result type or `None`:

```python
# readout_result.py
from typing import Generic, TypeVar

S = TypeVar("S", SamplingReadoutResult, None)
E = TypeVar("E", ExpectationReadoutResult, None)
T = TypeVar("T", StateTomographyReadoutResult, None)

@dataclass(frozen=True)
class ReadoutCompositeResults(Result, Generic[S, E, T]):
    sampling: S
    expectation_values: E
    state_tomography: T
```

When `S = SamplingReadoutResult`, the field type is `SamplingReadoutResult` — no `| None`, no guard needed. When `S = None`, accessing `.sampling.samples` is a type error because `None` has no attribute `samples`.

Construction becomes fully typed:

```python
# inside the backend, when building results for sampling + state_tomo:
results = ReadoutCompositeResults[SamplingReadoutResult, None, StateTomographyReadoutResult](
    sampling=sampling_result,
    expectation_values=None,
    state_tomography=state_tomo_result,
)
```

The `from_list` / `from_dict` constructors lose their raison d'etre — they exist to handle the untyped list case. Instead, the backend constructs the dataclass directly with typed fields.

### Layer 2: Generic `FunctionalResult`

Propagates the type parameters:

```python
# functional_result.py

class FunctionalResult(Result, Generic[S, E, T]):
    def __init__(
        self,
        readout_results: ReadoutCompositeResults[S, E, T],
        intermediate_results: list[ReadoutCompositeResults[S, E, T]] | None = None,
    ) -> None:
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results or []

    @property
    def readout_results(self) -> ReadoutCompositeResults[S, E, T]:
        return self._readout_results

    @property
    def intermediate_results(self) -> list[ReadoutCompositeResults[S, E, T]]:
        return self._intermediate_results
```

The convenience properties (`.samples`, `.state`, `.expected_values`) stay as runtime-checked shortcuts for notebook / REPL use. They don't participate in the generic type narrowing — but they don't need to, because the **typed safe path** goes through `.readout_results`:

```python
result.readout_results.sampling.samples           # fully typed, no guard
result.readout_results.state_tomography.state      # fully typed, no guard
result.readout_results.expectation_values          # type is None → .expected_values is a type error
```

The convenience shortcuts remain available for quick interactive work:

```python
result.samples          # works at runtime, type is dict[str, int] or raises ValueError
result.expected_values  # 💥 ValueError at runtime; the typed path caught it at write time
```

Two ergonomic tiers: quick (runtime-checked) and safe (statically-checked).

### Layer 3: Backend API — Keyword Arguments

This is the API change that makes it all work. Replace `readout: ReadoutMethod | list[ReadoutMethod]` with explicit keyword arguments:

```python
class Backend(ABC):

    # --- single readout overloads ---

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        sampling: SamplingReadout,
    ) -> FunctionalResult[SamplingReadoutResult, None, None]: ...

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        expectation: ExpectationReadout,
    ) -> FunctionalResult[None, ExpectationReadoutResult, None]: ...

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        state_tomography: StateTomographyReadout,
    ) -> FunctionalResult[None, None, StateTomographyReadoutResult]: ...

    # --- pairwise overloads ---

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        sampling: SamplingReadout,
        expectation: ExpectationReadout,
    ) -> FunctionalResult[SamplingReadoutResult, ExpectationReadoutResult, None]: ...

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        sampling: SamplingReadout,
        state_tomography: StateTomographyReadout,
    ) -> FunctionalResult[SamplingReadoutResult, None, StateTomographyReadoutResult]: ...

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        expectation: ExpectationReadout,
        state_tomography: StateTomographyReadout,
    ) -> FunctionalResult[None, ExpectationReadoutResult, StateTomographyReadoutResult]: ...

    # --- all three ---

    @overload
    def execute(
        self,
        functional: PrimitiveFunctional,
        *,
        sampling: SamplingReadout,
        expectation: ExpectationReadout,
        state_tomography: StateTomographyReadout,
    ) -> FunctionalResult[SamplingReadoutResult, ExpectationReadoutResult, StateTomographyReadoutResult]: ...

    # --- variational program (any readout combo) ---

    @overload
    def execute(
        self,
        functional: VariationalProgram,
        *,
        sampling: SamplingReadout | None = ...,
        expectation: ExpectationReadout | None = ...,
        state_tomography: StateTomographyReadout | None = ...,
    ) -> VariationalProgramResult: ...

    # --- implementation ---

    def execute(
        self,
        functional: Functional,
        *,
        sampling: SamplingReadout | None = None,
        expectation: ExpectationReadout | None = None,
        state_tomography: StateTomographyReadout | None = None,
    ) -> Result:
        readout_list: list[ReadoutMethod] = [
            ro for ro in (sampling, expectation, state_tomography) if ro is not None
        ]
        if not readout_list:
            raise ValueError("At least one readout method must be provided.")
        # ... dispatch to handler
```

7 overloads for `PrimitiveFunctional` (2^3 - 1) + 1 for `VariationalProgram` = 8 total. Verbose but mechanical, and it only changes when a new readout type is added.

### What This Gives the User

```python
# The type checker narrows the return type based on which kwargs are passed:

r1 = backend.execute(evolution, sampling=SamplingReadout(nshots=1000))
# Inferred: FunctionalResult[SamplingReadoutResult, None, None]

r1.readout_results.sampling.samples          # ✅ SamplingReadoutResult → dict[str, int]
r1.readout_results.sampling.probabilities    # ✅ SamplingReadoutResult → dict[str, float]
r1.readout_results.state_tomography          # type is None
r1.readout_results.state_tomography.state    # ❌ type error: None has no attribute 'state'
r1.readout_results.expectation_values        # type is None


r2 = backend.execute(
    evolution,
    sampling=SamplingReadout(nshots=1000),
    state_tomography=StateTomographyReadout(),
)
# Inferred: FunctionalResult[SamplingReadoutResult, None, StateTomographyReadoutResult]

r2.readout_results.sampling.samples          # ✅
r2.readout_results.state_tomography.state    # ✅
r2.readout_results.expectation_values        # None — accessing .expected_values is a type error
```

IDE autocomplete shows only the fields that exist. No `has_*` guards needed on the typed path. Misuse is caught before the code runs.

### What About the Convenience Properties? \*

The `.samples`, `.state`, `.expected_values` shortcuts on `FunctionalResult` can also participate in this narrowing. See the [Appendix: Typing the Convenience Properties](#appendix-typing-the-convenience-properties) for a full breakdown of three approaches (self-type method overloads, descriptors, and forwarding result objects) with tradeoffs.

### Alternative: Builder Pattern (Scales Better)

If you anticipate adding a 4th or 5th readout type, keyword arguments would need 2^N - 1 overloads. A builder avoids this:

```python
class ReadoutSpec(Generic[S, E, T]):
    """Type-safe specification of which readout methods to apply."""

    def __init__(self) -> None:
        self._sampling: SamplingReadout | None = None
        self._expectation: ExpectationReadout | None = None
        self._state_tomography: StateTomographyReadout | None = None

    def with_sampling(self, readout: SamplingReadout) -> ReadoutSpec[SamplingReadoutResult, E, T]:
        new = copy(self)
        new._sampling = readout
        return new  # type: ignore[return-value]

    def with_expectation(self, readout: ExpectationReadout) -> ReadoutSpec[S, ExpectationReadoutResult, T]:
        new = copy(self)
        new._expectation = readout
        return new  # type: ignore[return-value]

    def with_state_tomography(self, readout: StateTomographyReadout) -> ReadoutSpec[S, E, StateTomographyReadoutResult]:
        new = copy(self)
        new._state_tomography = readout
        return new  # type: ignore[return-value]
```

Usage:

```python
spec = (
    ReadoutSpec()
    .with_sampling(SamplingReadout(nshots=1000))
    .with_state_tomography(StateTomographyReadout())
)
# Inferred: ReadoutSpec[SamplingReadoutResult, None, StateTomographyReadoutResult]

result = backend.execute(evolution, spec)
# Inferred: FunctionalResult[SamplingReadoutResult, None, StateTomographyReadoutResult]
```

The backend needs only **two** overloads regardless of how many readout types exist:

```python
@overload
def execute(
    self, functional: PrimitiveFunctional, readout: ReadoutSpec[S, E, T]
) -> FunctionalResult[S, E, T]: ...

@overload
def execute(
    self, functional: VariationalProgram, readout: ReadoutSpec[S, E, T]
) -> VariationalProgramResult: ...
```

Each `with_*` method transforms one type parameter from `None` to the concrete result type. Adding a new readout means adding one type parameter and one `with_*` method — no overload explosion.

The tradeoff: slightly more verbose call site (`ReadoutSpec().with_sampling(...)` vs `sampling=...`). But it's chainable, discoverable via autocomplete, and scales to any number of readout types.

### Recommendation

For 3 readout types: **keyword arguments**. It's simpler, more Pythonic, and the 7 overloads are manageable.

If you foresee growing beyond 3-4 readout types: **builder pattern**. It scales with O(N) methods instead of O(2^N) overloads.

Either way, the core machinery is the same: `ReadoutCompositeResults[S, E, T]` and `FunctionalResult[S, E, T]` with constrained TypeVars.

---

## Appendix: Typing the Convenience Properties

The sections above establish that the **typed safe path** (`result.readout_results.sampling.samples`) is fully narrowed by the generic type parameters. But what about the convenience shortcuts — `result.samples`, `result.state`, `result.expected_values` — that live directly on `FunctionalResult`? Can those also participate in generic narrowing?

Yes. There are three approaches, each with different tradeoffs.

### Approach A: Self-Type Method Overloads

Use `@overload` where the discriminant is the type of `self`. Each property becomes a method with 2 overloads — one for "this readout is present" and one for "it's not":

```python
from typing import Any, Never, overload

class FunctionalResult(Result, Generic[S, E, T]):

    # .samples() — available when S = SamplingReadoutResult

    @overload
    def samples(self: FunctionalResult[SamplingReadoutResult, Any, Any]) -> dict[str, int]: ...
    @overload
    def samples(self: FunctionalResult[None, Any, Any]) -> Never: ...
    def samples(self) -> dict[str, int]:
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.samples
        raise ValueError("Sampling readout was not provided.")

    # .state() — available when T = StateTomographyReadoutResult

    @overload
    def state(self: FunctionalResult[Any, Any, StateTomographyReadoutResult]) -> QTensor: ...
    @overload
    def state(self: FunctionalResult[Any, Any, None]) -> Never: ...
    def state(self) -> QTensor:
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.state
        raise ValueError("State tomography readout was not provided.")

    # .expected_values() — available when E = ExpectationReadoutResult

    @overload
    def expected_values(self: FunctionalResult[Any, ExpectationReadoutResult, Any]) -> list[float]: ...
    @overload
    def expected_values(self: FunctionalResult[Any, None, Any]) -> Never: ...
    def expected_values(self) -> list[float]:
        if has_expectation_values(self._readout_results):
            return self._readout_results.expectation_values.expected_values
        raise ValueError("Expectation readout was not provided.")

    # .probabilities() — available when S OR T is present (3 overloads)

    @overload
    def probabilities(self: FunctionalResult[SamplingReadoutResult, Any, Any]) -> dict[str, float]: ...
    @overload
    def probabilities(self: FunctionalResult[None, Any, StateTomographyReadoutResult]) -> dict[str, float]: ...
    @overload
    def probabilities(self: FunctionalResult[None, Any, None]) -> Never: ...
    def probabilities(self) -> dict[str, float]:
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.probabilities
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.probabilities
        raise ValueError("Sampling or state tomography readout was not provided.")
```

That's 9 overloads total across 4 accessors. Each accessor needs only 2 overloads (present vs absent), except `.probabilities()` which draws from two sources (3 overloads). Not combinatorial at all.

At the call site:

```python
r = backend.execute(evolution, sampling=SamplingReadout(nshots=1000))
# Inferred: FunctionalResult[SamplingReadoutResult, None, None]

r.samples()          # ✅ type checker sees dict[str, int]
r.state()            # ❌ type checker sees Never — pyright flags this as unreachable/error
r.expected_values()  # ❌ type checker sees Never
```

**The catch**: these must be methods, not properties. The `@overload` decorator works on methods but not on `@property`. So the call site uses `result.samples()` instead of `result.samples`.

**Type checker support**: pyright handles self-type overloads well. mypy has partial/limited support (improving over time).

### Approach B: Descriptors With `__get__` Overloads

To keep the `result.samples` property syntax (no parentheses), define custom descriptors with overloaded `__get__`. Both mypy and pyright support `__get__` overloads on descriptors:

```python
from typing import Any, Never, overload


class _SamplesAccessor:
    @overload
    def __get__(
        self,
        obj: FunctionalResult[SamplingReadoutResult, Any, Any],
        objtype: type | None = None,
    ) -> dict[str, int]: ...

    @overload
    def __get__(
        self,
        obj: FunctionalResult[None, Any, Any],
        objtype: type | None = None,
    ) -> Never: ...

    def __get__(self, obj: Any, objtype: type | None = None) -> dict[str, int]:
        if obj is None:
            return self  # type: ignore[return-value]  # class-level access
        if has_sampling(obj._readout_results):
            return obj._readout_results.sampling.samples
        raise ValueError("Sampling readout was not provided.")


class _StateAccessor:
    @overload
    def __get__(
        self,
        obj: FunctionalResult[Any, Any, StateTomographyReadoutResult],
        objtype: type | None = None,
    ) -> QTensor: ...

    @overload
    def __get__(
        self,
        obj: FunctionalResult[Any, Any, None],
        objtype: type | None = None,
    ) -> Never: ...

    def __get__(self, obj: Any, objtype: type | None = None) -> QTensor:
        if obj is None:
            return self  # type: ignore[return-value]
        if has_state_tomography(obj._readout_results):
            return obj._readout_results.state_tomography.state
        raise ValueError("State tomography readout was not provided.")


class _ExpectedValuesAccessor:
    @overload
    def __get__(
        self,
        obj: FunctionalResult[Any, ExpectationReadoutResult, Any],
        objtype: type | None = None,
    ) -> list[float]: ...

    @overload
    def __get__(
        self,
        obj: FunctionalResult[Any, None, Any],
        objtype: type | None = None,
    ) -> Never: ...

    def __get__(self, obj: Any, objtype: type | None = None) -> list[float]:
        if obj is None:
            return self  # type: ignore[return-value]
        if has_expectation_values(obj._readout_results):
            return obj._readout_results.expectation_values.expected_values
        raise ValueError("Expectation readout was not provided.")


class _ProbabilitiesAccessor:
    @overload
    def __get__(
        self,
        obj: FunctionalResult[SamplingReadoutResult, Any, Any],
        objtype: type | None = None,
    ) -> dict[str, float]: ...

    @overload
    def __get__(
        self,
        obj: FunctionalResult[None, Any, StateTomographyReadoutResult],
        objtype: type | None = None,
    ) -> dict[str, float]: ...

    @overload
    def __get__(
        self,
        obj: FunctionalResult[None, Any, None],
        objtype: type | None = None,
    ) -> Never: ...

    def __get__(self, obj: Any, objtype: type | None = None) -> dict[str, float]:
        if obj is None:
            return self  # type: ignore[return-value]
        if has_sampling(obj._readout_results):
            return obj._readout_results.sampling.probabilities
        if has_state_tomography(obj._readout_results):
            return obj._readout_results.state_tomography.probabilities
        raise ValueError("Sampling or state tomography readout was not provided.")


class FunctionalResult(Result, Generic[S, E, T]):
    samples = _SamplesAccessor()
    state = _StateAccessor()
    expected_values = _ExpectedValuesAccessor()
    probabilities = _ProbabilitiesAccessor()

    def __init__(
        self,
        readout_results: ReadoutCompositeResults[S, E, T],
        intermediate_results: list[ReadoutCompositeResults[S, E, T]] | None = None,
    ) -> None:
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results or []

    @property
    def readout_results(self) -> ReadoutCompositeResults[S, E, T]:
        return self._readout_results
```

Now the user writes `result.samples` (property syntax, no parentheses), and the type checker sees either `dict[str, int]` or `Never` depending on the generic parameter:

```python
r = backend.execute(evolution, sampling=SamplingReadout(nshots=1000))
# Inferred: FunctionalResult[SamplingReadoutResult, None, None]

r.samples              # ✅ dict[str, int]
r.state                # ❌ Never — type error
r.expected_values      # ❌ Never — type error
```

**Tradeoffs**: One descriptor class (~15-20 lines) per property. Mechanical but verbose. Works with both mypy and pyright.

### Approach C: Forwarding Result Objects (Simplest)

Instead of unwrapping `.samples` / `.state` / `.expected_values` from result objects, forward the result objects themselves as typed properties:

```python
class FunctionalResult(Result, Generic[S, E, T]):

    @property
    def sampling(self) -> S:
        return self._readout_results.sampling

    @property
    def expectation(self) -> E:
        return self._readout_results.expectation_values

    @property
    def state_tomography(self) -> T:
        return self._readout_results.state_tomography
```

No overloads, no descriptors, no tricks. The generic type parameter does all the work:

```python
r = backend.execute(evolution, sampling=SamplingReadout(1000))
# Inferred: FunctionalResult[SamplingReadoutResult, None, None]

r.sampling                  # SamplingReadoutResult (not Optional!)
r.sampling.samples          # ✅ dict[str, int]
r.sampling.probabilities    # ✅ dict[str, float]
r.state_tomography          # None
r.state_tomography.state    # ❌ type error: None has no attribute 'state'
r.expectation               # None
```

The cost is one extra attribute access (`result.sampling.samples` instead of `result.samples`). But:
- Zero overloads, zero descriptors, zero boilerplate
- Fully typed with no escape hatches
- The result objects carry richer APIs (`.get_probability()`, `.get_probabilities()`) that are also typed and discoverable
- Mirrors the structure of the readout spec — what you asked for is what you get

### Comparison

| Approach | Syntax | Type safety | mypy support | pyright support | Boilerplate |
|---|---|---|---|---|---|
| **A.** Self-type method overloads | `result.samples()` | Full | Partial | Yes | ~6 lines per accessor |
| **B.** Descriptors with `__get__` overloads | `result.samples` | Full | Yes | Yes | ~20 lines per accessor |
| **C.** Forwarding result objects | `result.sampling.samples` | Full | Yes | Yes | ~3 lines per accessor |

**Recommendation**: Approach C (forwarding) gives full type safety with essentially zero machinery. If the team strongly prefers `result.samples` over `result.sampling.samples`, use Approach B (descriptors) — they're verbose but they work everywhere. Approach A (self-type overloads) is the middle ground but forces methods instead of properties and has weaker mypy support.
