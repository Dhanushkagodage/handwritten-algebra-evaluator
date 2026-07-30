"""
Run all ten sample cases back-to-back in a single process.

The individual tests/test_case_NN.py scripts are the normal way to work —
one case, one Colab cell. But on the `local` backend each of those starts a
fresh process and reloads Qwen2.5-3B from scratch (several minutes each).
This script builds the model once and reuses it for all ten, and prints a
pass/fail table at the end.

    python tests/run_all_cases.py --backend local   # what this is for
    python tests/run_all_cases.py --cases 2,5,9     # a subset

Careful with --backend space here: ten Space calls in quick succession will
eat a visible chunk of your ZeroGPU quota. Prefer `local` for the full sweep.

Prints only — nothing is written to disk.
"""
import asyncio
import sys

from case_runner import make_generator, resolve_backend, run_case

ALL_CASES = tuple(range(1, 11))


def resolve_cases(argv=None) -> tuple:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cases" not in argv:
        return ALL_CASES

    i = argv.index("--cases")
    if i + 1 >= len(argv):
        raise SystemExit("--cases needs a value, e.g. --cases 2,5,9")

    cases = []
    for part in argv[i + 1].split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) not in ALL_CASES:
            raise SystemExit(f"unknown case {part!r} — valid cases are 1-10")
        cases.append(int(part))
    if not cases:
        raise SystemExit("--cases needs at least one case number")
    return tuple(cases)


async def run() -> int:
    backend = resolve_backend()
    cases = resolve_cases()
    print(f"Backend : {backend}")
    print(f"Cases   : {', '.join(f'{c:02d}' for c in cases)}\n")

    generator = make_generator(backend)
    await generator.load_model()

    results = {}
    for case_number in cases:
        try:
            results[case_number] = await run_case(case_number, generator)
        except Exception as exc:
            # One bad case shouldn't cost us the other nine, especially on the
            # space backend where a cold Space can time out.
            print(f"\nCase {case_number:02d} raised: {type(exc).__name__}: {exc}")
            results[case_number] = [f"{type(exc).__name__}: {exc}"]
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for case_number in cases:
        failures = results[case_number]
        if failures:
            print(f"  case {case_number:02d}  FAIL  {failures[0]}")
            for extra in failures[1:]:
                print(f"                  {extra}")
        else:
            print(f"  case {case_number:02d}  PASS")

    passed = sum(1 for c in cases if not results[c])
    print(f"\n{passed}/{len(cases)} passed  (backend: {backend})")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
