"""Stage 22 M2 — concept-level rules with bounded unification-lite proving.

Extends the M1 hierarchy into deduction: rules quantified over concepts
(``?vars``), proved by bounded backward chaining with variable binding
(the SymbolBackend's wildcard matching has no binding; this module adds
a small, bounded unifier for concept-level queries).

Built-in concept rules (when the hierarchy bridge is used):
  R1  member(?x,?c) ∧ prop(?c,?p)          -> prop(?x,?p)      inheritance
  R2  member(?x,?c) ∧ parent(?c,?s)        -> member(?x,?s)    abstraction
  R3  prop(?x,?p)   ∧ implies(?p,?q)       -> prop(?x,?q)      rule composition

M2 verification (docs/stage22_design.md): multi-hop concept reasoning
batteries (2-hop via the super level, 3-hop via implies) pass at high
rates, while the instance-level baseline (R1/R3 only, no abstraction)
cannot join the parent chain by construction — with SOUNDNESS:
unsupported queries must return empty (no fabrication).

Bounded by construction: facts <= max_facts, rules <= max_rules,
answers <= max_answers, recursion depth <= max_depth, op budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Rule:
    antecedents: list[tuple[str, tuple]]
    consequent: tuple[str, tuple]


class ConceptProver:
    """Bounded backward-chaining prover with unification-lite (``?vars``)."""

    def __init__(self, max_facts: int = 1024, max_rules: int = 128,
                 max_answers: int = 64, max_depth: int = 3,
                 op_budget: int = 20000) -> None:
        self.max_facts = int(max_facts)
        self.max_rules = int(max_rules)
        self.max_answers = int(max_answers)
        self.max_depth = int(max_depth)
        self.op_budget = int(op_budget)
        self._facts: dict[str, set[tuple]] = {}
        self._rules: list[_Rule] = []
        self._n_facts = 0
        self.last_ops = 0  # observability for the op budget

    # ------------------------------------------------------------- ingestion

    def add_fact(self, pred: str, args: tuple) -> bool:
        if self._n_facts >= self.max_facts:
            return False
        self._facts.setdefault(pred, set()).add(tuple(args))
        self._n_facts += 1
        return True

    def add_rule(self, antecedents: list[tuple[str, tuple]],
                 consequent: tuple[str, tuple]) -> bool:
        if len(self._rules) >= self.max_rules:
            return False
        # 2026-10-11: rename rule variables into a per-rule namespace
        # (``?x`` -> ``?r{idx}_x``). Name-based bindings otherwise collide a
        # rule's local vars with the goal's/other rules' vars ("?c" reuse
        # across R1/R2 silently failed the 2-hop chain).
        idx = len(self._rules)

        def _rn(pred: str, args: tuple) -> tuple[str, tuple]:
            out = []
            for a in args:
                if self._is_var(a):
                    out.append(f"?r{idx}_{a[1:]}")
                else:
                    out.append(a)
            return pred, tuple(out)

        ants = [_rn(p, a) for p, a in antecedents]
        cons = _rn(consequent[0], consequent[1])
        self._rules.append(_Rule(ants, cons))
        return True

    def add_concept_rules(self) -> None:
        """Inject the standard concept rules (R1-R3)."""
        self.add_rule([("member", ("?x", "?c")), ("prop", ("?c", "?p"))],
                      ("prop", ("?x", "?p")))
        self.add_rule([("member", ("?x", "?c")), ("parent", ("?c", "?s"))],
                      ("member", ("?x", "?s")))
        self.add_rule([("prop", ("?x", "?p")), ("implies", ("?p", "?q"))],
                      ("prop", ("?x", "?q")))

    # --------------------------------------------------------------- prover

    @staticmethod
    def _is_var(t) -> bool:
        return isinstance(t, str) and t.startswith("?")

    @staticmethod
    def _deref(bindings: dict, t):
        """Follow bindings (var -> const or var) to the base term."""
        seen = set()
        while ConceptProver._is_var(t) and t in bindings and t not in seen:
            seen.add(t)
            t = bindings[t]
        return t

    @classmethod
    def _unify(cls, pat_args: tuple, fact_args: tuple,
               bindings: dict) -> dict | None:
        """Unify a pattern with a fact/term; var-var aliases are linked."""
        if len(pat_args) != len(fact_args):
            return None
        b = dict(bindings)
        for p, f in zip(pat_args, fact_args):
            if p == "_":
                continue
            pv = cls._deref(b, p) if cls._is_var(p) else p
            fv = cls._deref(b, f) if cls._is_var(f) else f
            if cls._is_var(pv) and cls._is_var(fv):
                if pv != fv:
                    b[pv] = fv
            elif cls._is_var(pv):
                b[pv] = fv
            elif cls._is_var(fv):
                b[fv] = pv
            elif pv != fv:
                return None
        return b

    def _resolve(self, args: tuple, bindings: dict) -> tuple | None:
        out = []
        for a in args:
            v = self._deref(bindings, a) if self._is_var(a) else a
            out.append(v)
        return tuple(out)

    def query(self, pred: str, args: tuple) -> list[tuple]:
        """Return derivable arg-tuples for the query (bounded, sound).

        Backward chaining with unification-lite: facts first, then rules
        whose consequent unifies with the goal; antecedents are solved
        sequentially (conjunction) with bindings threaded through.
        """
        answers: list[tuple] = []
        self._ops = 0
        goal_args = tuple(args)

        def solve(g_pred: str, g_args: tuple, bindings: dict,
                  depth: int):
            """Yield extended bindings satisfying the goal."""
            if self._ops > self.op_budget or len(answers) >= self.max_answers:
                return
            self._ops += 1
            for fact in self._facts.get(g_pred, ()):  # BOUNDS-OK bounded store
                b = self._unify(g_args, fact, bindings)
                if b is not None:
                    yield b
            if depth >= self.max_depth:
                return
            for rule in self._rules:
                c_pred, c_args = rule.consequent
                if c_pred != g_pred:
                    continue
                b0 = self._unify(c_args, g_args, bindings)
                if b0 is None:
                    continue
                # conjunction of antecedents, bound sequentially
                stack = [(0, b0)]
                while stack:
                    if self._ops > self.op_budget or len(answers) >= self.max_answers:
                        return
                    i, b = stack.pop()
                    if i == len(rule.antecedents):
                        yield b
                        continue
                    ap, aa = rule.antecedents[i]
                    for b2 in solve(ap, aa, b, depth + 1):
                        stack.append((i + 1, b2))

        for b in solve(pred, goal_args, {}, 0):
            # answers = the bound values at the query's VARIABLE positions
            # (ground queries answer by existence: the full tuple).
            var_pos = [i for i, a in enumerate(goal_args) if self._is_var(a)]
            if var_pos:
                r = tuple(self._deref(b, goal_args[i]) for i in var_pos)
            else:
                r = tuple(goal_args)
            if r not in answers:
                answers.append(r)
                if len(answers) >= self.max_answers:
                    break
        self.last_ops = self._ops
        return answers

    # ---------------------------------------------------------------- stats

    @property
    def n_facts(self) -> int:
        return self._n_facts

    @property
    def n_rules(self) -> int:
        return len(self._rules)
