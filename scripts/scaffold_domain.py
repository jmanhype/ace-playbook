"""Scaffold helper for creating a new ACE benchmark/guardrail domain.

Generates:
  * benchmarks/<domain>.jsonl            (sample benchmark file)
  * ace/utils/<domain>_guardrails.py     (guardrail module template)
  * docs/domains/<domain>.rst            (docs stub)

Usage:
    python scripts/scaffold_domain.py finance-lite

Note: This script intentionally creates minimal stubs; you must fill in
ground-truth entries and guardrail calculators manually.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parent.parent


BENCHMARK_TEMPLATE = Template(
    dedent(
        """\
        {
            "task_id": "${domain}-001",
            "description": "TODO: describe the task input",
            "ground_truth": "TODO"
        }
        """
    ).strip()
)


GUARDRAIL_TEMPLATE = Template(
    dedent(
        '''\
        """Guardrails for ${domain} benchmark tasks."""

        from __future__ import annotations

        from dataclasses import dataclass
        from decimal import Decimal
        from typing import Callable, Dict, Optional


        DecimalCalculator = Callable[[], Decimal]


        @dataclass(frozen=True)
        class DomainGuardrail:
            instructions: str
            calculator: Optional[DecimalCalculator] = None
            auto_correct: bool = False

            def canonical_answer(self) -> Optional[str]:
                if not self.calculator:
                    return None
                return str(self.calculator())


        DOMAIN_GUARDRAILS: Dict[str, DomainGuardrail] = {
            "${domain}-001": DomainGuardrail(
                instructions="TODO: describe how to compute the answer",
                calculator=None,
                auto_correct=False,
            ),
        }


        def get_guardrail(task_id: str) -> Optional[DomainGuardrail]:
            return DOMAIN_GUARDRAILS.get(task_id)
        '''
    ).strip()
)


DOCS_TEMPLATE = Template(
    """$title
$underline

Overview
--------

.. todo:: Describe the domain, ground-truth source, and guardrails.

Benchmark
---------

* Benchmark file: ``benchmarks/$domain.jsonl``
* Guardrails: ``ace/utils/${domain}_guardrails.py``
* Results: ``results/ace_full_${domain}.json``

Setup Steps
-----------

1. Populate ``benchmarks/$domain.jsonl`` with real tasks and ground truth.
2. Implement guardrail calculators/validators.
3. Run ``python scripts/run_benchmark.py benchmarks/$domain.jsonl ace_full --output results/ace_full_${domain}.json``.
4. Document any special considerations here.

"""
)


def write_file(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def scaffold_domain(domain: str) -> None:
    domain = domain.lower().replace(" ", "-")
    benchmark_path = PROJECT_ROOT / "benchmarks" / f"{domain}.jsonl"
    guardrail_path = PROJECT_ROOT / "ace" / "utils" / f"{domain}_guardrails.py"
    docs_path = PROJECT_ROOT / "docs" / "domains" / f"{domain}.rst"

    write_file(benchmark_path, BENCHMARK_TEMPLATE.substitute(domain=domain))
    write_file(guardrail_path, GUARDRAIL_TEMPLATE.substitute(domain=domain))

    title = f"{domain.replace('-', ' ').title()} Domain"
    underline = "=" * len(title)
    write_file(
        docs_path,
        DOCS_TEMPLATE.substitute(domain=domain, title=title, underline=underline),
    )

    print(f"Created benchmark stub: {benchmark_path}")
    print(f"Created guardrail stub: {guardrail_path}")
    print(f"Created docs stub: {docs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new ACE domain template")
    parser.add_argument("domain", help="Domain name (e.g., finance-lite)")
    args = parser.parse_args()
    scaffold_domain(args.domain)


if __name__ == "__main__":
    main()
