from typing import Generic, TypeVar, overload

from qilisdk.functionals import Sampling, SamplingResult, TimeEvolution, TimeEvolutionResult
from qilisdk.functionals.functional import Functional
from qilisdk.functionals.functional_result import FunctionalResult

TResult = TypeVar("TResult", bound=FunctionalResult)


class JobHandle(Generic[TResult]):
    # Keep the runtime class of the result so get_job can build the right object.
    def __init__(self, id: int, result_type: type[TResult]) -> None:
        self.id = id
        self.result_type = result_type


class Job(Generic[TResult]):
    def __init__(self, id: int, result: TResult) -> None:
        self.id = id
        self._result = result

    def get_results(self) -> TResult:
        return self._result


# --- submit_job preserves the concrete TResult on the handle ------------------

@overload
def submit_job(functional: TimeEvolution) -> JobHandle[TimeEvolutionResult]: ...
@overload
def submit_job(functional: Sampling) -> JobHandle[SamplingResult]: ...
def submit_job(functional: Functional[FunctionalResult]) -> JobHandle[FunctionalResult]:
    # real code would submit and return the new job id
    job_id = 10
    if isinstance(functional, TimeEvolution):
        return JobHandle(job_id, TimeEvolutionResult)
    else:
        return JobHandle(job_id, SamplingResult)


# --- get_job returns a Job parametrized by the same TResult as the handle -----

@overload
def get_job(job_handle: JobHandle[TimeEvolutionResult]) -> Job[TimeEvolutionResult]: ...
@overload
def get_job(job_handle: JobHandle[SamplingResult]) -> Job[SamplingResult]: ...
def get_job(job_handle: JobHandle[FunctionalResult]) -> Job[FunctionalResult]:
    # pretend we fetched a payload; choose the right result class from the handle
    if job_handle.result_type is TimeEvolutionResult:
        # Build from whatever your backend returns:
        result = TimeEvolutionResult(...)  # e.g., TimeEvolutionResult.from_payload(payload)
        return Job(job_handle.id, result)
    else:
        result = SamplingResult(...)
        return Job(job_handle.id, result)


# ---------------------------- usage (type inference) --------------------------
te_handle = submit_job(TimeEvolution(...))      # JobHandle[TimeEvolutionResult]
te_job = get_job(te_handle)                     # Job[TimeEvolutionResult]
te_res = te_job.get_results()                   # TimeEvolutionResult

sa_handle = submit_job(Sampling(...))           # JobHandle[SamplingResult]
sa_job = get_job(sa_handle)                     # Job[SamplingResult]
sa_res = sa_job.get_results()                   # SamplingResult
