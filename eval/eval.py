"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import asyncio, os, json, time as t, yaml, hashlib, random, logging

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DEBUG"] = "False"
from argparse import ArgumentParser
from typing import List, Dict, Any, Optional, Tuple
from litellm import acompletion
import re
import litellm

litellm.drop_params = True
litellm._turn_on_debug()

# Logging
h = logging.StreamHandler()
h.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
log = logging.getLogger("runner")
log.setLevel(logging.INFO)
log.handlers[:] = [h]


# Model
class AsyncModel:
    def __init__(
        self,
        name: str,
        greedy: bool = False,
        port: Optional[int] = None,
        host: str = "localhost",
    ):
        self.endpoint, self.kw = self._resolve(name, greedy, port, host)

    @staticmethod
    def _resolve(
        name: str, greedy: bool, port: Optional[int], host: str = "localhost"
    ) -> Tuple[str, Dict[str, Any]]:
        api_base = api_key = api_version = None
        if name == "gpt-4o":
            ep = "azure/gpt-4o"
            api_base = os.getenv("AZURE_API_BASE_gpt_4o")
            api_key = os.getenv("AZURE_API_KEY_gpt_4o")
            api_version = os.getenv("AZURE_API_VERSION_gpt_4o")
        elif name == "o3":
            ep = "azure/o3"
            api_base = os.getenv("AZURE_API_BASE_o3")
            api_key = os.getenv("AZURE_API_KEY_o3")
            api_version = os.getenv("AZURE_API_VERSION_o3")
        elif name == "gpt-5":
            ep = "azure/gpt-5"
            api_base = os.getenv("AZURE_API_BASE_gpt_5")
            api_key = os.getenv("AZURE_API_KEY_gpt_5")
            api_version = os.getenv("AZURE_API_VERSION_gpt_5")
        elif "gemini" in name:
            ep = "gemini/gemini-2.5-flash-preview-05-20"
        elif "claude" in name:
            ep = f"bedrock/{name}"
        else:
            ep = f"hosted_vllm/{name}"
            api_base = f"http://{host}:{port}/v1"
        kw = {"timeout": 300, "num_retries": 5}
        if api_base:
            kw["api_base"] = api_base
        if api_key:
            kw["api_key"] = api_key
        if api_version:
            kw["api_version"] = api_version
        if greedy:
            kw["temperature"] = 0
        return ep, kw

    async def raw(self, prompt: str):
        return await acompletion(
            model=self.endpoint,
            messages=[{"role": "user", "content": prompt}],
            **self.kw,
        )


# Cost Tracking
class CostTracker:
    def __init__(self):
        self.calls = 0
        self.failures = 0
        self.total_cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _num(x):
        try:
            return int(x)
        except:
            return 0

    @staticmethod
    def _flt(x):
        try:
            return float(x)
        except:
            return 0.0

    async def add_success(self, resp) -> None:
        cost = 0.0
        try:
            cost = self._flt(getattr(resp, "_hidden_params", {}).get("response_cost"))
        except Exception:
            pass
        usage = getattr(resp, "usage", {}) or {}
        pt = self._num(usage.get("prompt_tokens") or usage.get("input_tokens"))
        ct = self._num(usage.get("completion_tokens") or usage.get("output_tokens"))
        tt = self._num(usage.get("total_tokens") or (pt + ct))
        async with self._lock:
            self.calls += 1
            self.total_cost += cost
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_tokens += tt

    async def add_failure(self) -> None:
        async with self._lock:
            self.failures += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "cost_usd": round(self.total_cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# Progress Logging
class Progress:
    def __init__(self, costs: CostTracker, interval_s: float = 5.0):
        self.start = t.monotonic()
        self.total = 0
        self.done = 0
        self.costs = costs
        self._last = 0.0
        self._interval = interval_s
        self._lock = asyncio.Lock()

    async def set_total(self, n: int):
        async with self._lock:
            self.total = n

    async def tick(self, n: int = 1):
        async with self._lock:
            self.done += n
        self.maybe_log()

    @staticmethod
    def _fmt(secs: Optional[float]):
        if secs is None:
            return "n/a"
        s = max(0, int(secs))
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def snap(self):
        el = t.monotonic() - self.start
        if self.done and self.total:
            rate = self.done / el if el > 0 else 0
            est_total = (self.total / rate) if rate > 0 else None
            eta = (est_total - el) if est_total else None
        else:
            est_total = eta = None
        return dict(
            done=self.done,
            total=self.total,
            elapsed=self._fmt(el),
            est_total=self._fmt(est_total),
            eta=self._fmt(eta),
        )

    def maybe_log(self):
        now = t.monotonic()
        if now - self._last >= self._interval:
            self._last = now
            s = self.snap()
            c = self.costs.snapshot()
            log.info(
                f"[ETA] {s['done']}/{s['total']} | elapsed {s['elapsed']} | "
                f"total~ {s['est_total']} | ETA {s['eta']} | "
                f"cost ${c['cost_usd']:.6f} | calls {c['calls']} (fail {c['failures']})"
            )


# Helpers
def load_profiles(path: str, limit: int):
    raw = open(path).read().strip()
    try:
        obj = json.loads(raw)
        data = obj if isinstance(obj, list) else obj.get("profiles", [obj])
    except json.JSONDecodeError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    return data[:limit]


def load_prompts(path: str, level: int):
    p = yaml.safe_load(open(path))
    return {
        "task_solving": p["task_solving"][int(level)],
        "adversary_reveal": p["adversary_reveal"][1],
    }


def info_attr_names(profile: Dict[str, Any]) -> List[str]:
    return list((profile["information_attributes"]).keys())


def memory_map(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    mem: Dict[str, List[str]] = {}
    bio = profile.get("bio") or {}
    if "name" in bio:
        mem.setdefault("name", []).append(f"{bio['name']} is the person's name.")
    if "age" in bio:
        mem.setdefault("age", []).append(f"The person is {bio['age']} years old.")
    if "sex" in bio:
        mem.setdefault("sex", []).append(f"The person's sex is {bio['sex']}.")
    if "address" in bio:
        mem.setdefault("address", []).append(
            f"The person's address is {bio['address']}."
        )
    for k, v in bio.items():
        if k not in {"name", "age", "sex", "address"}:
            mem.setdefault(k, []).append(f"{k}: {v}")
    for a, obj in (profile.get("information_attributes") or {}).items():
        if isinstance(obj, dict):
            ms = obj.get("memory_statement")
            val = obj.get("value")
            if isinstance(ms, str) and ms.strip():
                mem.setdefault(a, []).append(ms.strip())
            elif val is not None:
                mem.setdefault(a, []).append(f"{a}: {val}")
        else:
            mem.setdefault(a, []).append(f"{a}: {obj}")
    return mem


def attr_lines(names: List[str], mem: Dict[str, List[str]]) -> List[str]:
    return [f"{a} → {mem[a][0]}" if mem.get(a) else a for a in names]


def mem_blob(attrs: List[str], mem: Dict[str, List[str]]) -> str:
    return "\n".join([mem[a][0] for a in attrs if mem.get(a)]).strip()


def parse_list(text: str) -> List[str]:
    s = (text or "").replace("```", "").strip()
    parts = [x.strip() for x in s.split("\n") if x.strip()]
    return parts if len(parts) > 1 else [x.strip() for x in s.split(",") if x.strip()]


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
    except Exception as e:
        print("Failed to decode JSON response:", e)
        return {}


def combos_full(
    A: List[str], B: List[str], C: List[str], trials: int
) -> List[Tuple[int, List[str], List[str]]]:
    out = []
    assert len(B) == 0 and len(C) == 0
    for tr in range(trials):
        out.append((tr, list(A), list(B), list(C)))
    return out


def combos_iid(
    A: List[str], B: List[str], C: List[str], trials: int
) -> List[Tuple[int, List[str], List[str], List[str]]]:
    out = []
    for mem_size in range(1, len(A) + len(B) + 1, 10):
        memories = random.sample(A + B, mem_size)
        A_ = [a for a in memories if a in A]
        B_ = [b for b in memories if b in B]
        for tr in range(trials):
            out.append((tr, A_, B_, C))
    return out  # = (|A|+|B|)*trials


def combos_all_share(
    A: List[str], B: List[str], C: List[str], trials: int
) -> List[Tuple[int, List[str], List[str]]]:
    out = []
    for num_not_to_share in range(0, len(B) + 1, 10):
        random_sample_not_to_share = (
            random.sample(B, num_not_to_share) if num_not_to_share > 0 else []
        )
        for tr in range(trials):
            out.append((tr, list(A), list(random_sample_not_to_share), list(C)))
    return out  # = (|B|+1)*trials


def combos_all_not_share(
    A: List[str], B: List[str], C: List[str], trials: int
) -> List[Tuple[int, List[str], List[str]]]:
    out = []
    for num_to_share in range(0, len(A) + 1, 1):
        random_sample_to_share = (
            random.sample(A, num_to_share) if num_to_share > 0 else []
        )
        for tr in range(trials):
            out.append((tr, list(random_sample_to_share), list(B), list(C)))
    return out  # = (|A|+1)*trials


async def run_calls(
    prompts: List[str],
    model: AsyncModel,
    sem: asyncio.Semaphore,
    prog: Progress,
    costs: CostTracker,
    label: str,
) -> List[str]:
    res: List[Optional[str]] = [None] * len(prompts)

    async def one(i: int, p: str):
        try:
            async with sem:
                resp = await model.raw(p)
        except Exception as e:
            log.error(f"{label}@{i}: {e}")
            await costs.add_failure()
            res[i] = ""
            await prog.tick(1)
            return
        try:
            txt = resp.choices[0].message.content
        except Exception:
            txt = ""
        await costs.add_success(resp)
        res[i] = txt or ""
        await prog.tick(1)

    if prompts:
        await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])
    return [r or "" for r in res]


# Main Pipeline
async def amain(a):
    log.info(
        f"model={a.model_name} strong={a.strong_model_name} concurrency={a.concurrency}"
    )
    weak = AsyncModel(
        a.model_name, greedy=a.greedy, port=a.model_port, host=a.model_host
    )
    strong = AsyncModel(
        a.strong_model_name,
        greedy=True,
        port=a.strong_model_port,
        host=a.strong_model_host,
    )

    # health check
    try:
        await weak.raw("Just say hi!")
        await strong.raw("Just say hi!")
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return
    log.info("Health check passed")

    data = load_profiles(a.data_file, a.num_profiles)
    P = load_prompts(a.prompts_file, a.privacy_prompts_level)

    # plan contexts
    plan = []
    for prof in data:
        names = info_attr_names(prof)
        mem = memory_map(prof)
        for ctx in prof.get("contexts") or []:
            plan.append(
                {
                    "prof": prof,
                    "ctx": ctx,
                    "names": names,
                    "mem": mem,
                    "A": [],
                    "B": [],
                    "combos": [],
                }
            )

    sem = asyncio.Semaphore(max(1, a.concurrency))
    costs = CostTracker()
    prog = Progress(costs, a.log_interval)

    await prog.set_total(len(plan))

    total_combo = 0
    for r in plan:
        A = [
            attr
            for attr in r["names"]
            if max(
                r["ctx"]["labels_combined"][
                    r["prof"]["information_attributes"][attr]["memory_statement"]
                ],
                key=r["ctx"]["labels_combined"][
                    r["prof"]["information_attributes"][attr]["memory_statement"]
                ].get,
            )
            == "necessary"
            and r["ctx"]["labels_combined"][
                r["prof"]["information_attributes"][attr]["memory_statement"]
            ]["necessary"]
            == 1
        ]
        B = [
            attr
            for attr in r["names"]
            if max(
                r["ctx"]["labels_combined"][
                    r["prof"]["information_attributes"][attr]["memory_statement"]
                ],
                key=r["ctx"]["labels_combined"][
                    r["prof"]["information_attributes"][attr]["memory_statement"]
                ].get,
            )
            == "private"
            and r["ctx"]["labels_combined"][
                r["prof"]["information_attributes"][attr]["memory_statement"]
            ]["private"]
            == 1
        ]
        C = [attr for attr in r["names"] if attr not in A and attr not in B]

        r["A"], r["B"], r["C"] = A, B, C
        if a.combo_style == "stratified":
            raise NotImplementedError()
        elif a.combo_style == "full":
            total_combo += a.num_trials
        elif a.combo_style == "iid":
            total_combo += len(combos_iid(r["A"], r["B"], r["C"], a.num_trials))
        elif a.combo_style == "all_share":
            total_combo += len(combos_all_share(r["A"], r["B"], r["C"], a.num_trials))
        elif a.combo_style == "all_not_share":
            total_combo += len(
                combos_all_not_share(r["A"], r["B"], r["C"], a.num_trials)
            )
        else:
            raise ValueError(f"Unknown combo style {a.combo_style}")

    final_total = len(plan) + 2 * total_combo
    await prog.set_total(final_total)
    log.info(
        f"Stage planning: combos={total_combo} planned_calls={final_total} | costs_so_far={costs.snapshot()}"
    )

    # Stage A: tasks
    task_prompts, task_refs = [], []
    for r in plan:
        if a.combo_style == "stratified":
            raise NotImplementedError()
        elif a.combo_style == "full":
            r["combos"] = combos_full(r["names"], [], [], a.num_trials)
        elif a.combo_style == "iid":
            r["combos"] = combos_iid(r["A"], r["B"], r["C"], a.num_trials)
        elif a.combo_style == "all_share":
            r["combos"] = combos_all_share(r["A"], r["B"], r["C"], a.num_trials)
        elif a.combo_style == "all_not_share":
            r["combos"] = combos_all_not_share(r["A"], r["B"], r["C"], a.num_trials)
        else:
            raise ValueError(f"Unknown combo_style: {a.combo_style}")

        for tr, A_, B_, C_ in r["combos"]:
            task_prompt = P["task_solving"].format(
                task=r["ctx"].get("task", ""),
                recipient=(r["ctx"].get("recipient") or "").lower(),
                memories=mem_blob(
                    [a for a in r["names"] if a in A_ + B_ + C_], r["mem"]
                ),
            )
            task_prompts.append(task_prompt)
            k = f"combination_num_to_share={len(A_)}_num_not_to_share={len(B_)}_trial={tr}"
            r["ctx"].setdefault(k, {})["prompt"] = task_prompt
            task_refs.append((r["ctx"], tr, A_, B_, C_))
    log.info(f"Stage A: {len(task_prompts)} task calls")
    task_out = await run_calls(task_prompts, weak, sem, prog, costs, "task")

    for (ctx, tr, A_, B_, C_), sol in zip(task_refs, task_out):
        k = f"combination_num_to_share={len(A_)}_num_not_to_share={len(B_)}_trial={tr}"
        b = ctx.setdefault(k, {})
        b.update(
            {
                "to_share_attributes": A_,
                "not_to_share_attributes": B_,
                "num_to_share_attributes": len(A_),
                "num_not_to_share_attributes": len(B_),
                "total_num_attributes": len(A_) + len(B_),
                "trial_idx": tr,
            }
        )
        b.setdefault("model_solution", {})["response_solution"] = (sol or "").strip()

    log.info(f"Stage A complete | costs={costs.snapshot()}")

    # Stage B: reveals
    rev_prompts, rev_refs = [], []
    for r in plan:
        for tr, A_, B_, C_ in r["combos"]:
            k = f"combination_num_to_share={len(A_)}_num_not_to_share={len(B_)}_trial={tr}"
            sol = ((r["ctx"].get(k) or {}).get("model_solution") or {}).get(
                "response_solution"
            ) or ""
            rev_prompts.append(
                P["adversary_reveal"].format(
                    attributes="\n".join(attr_lines(A_ + B_ + C_, r["mem"])),
                    solution=sol,
                )
            )
            rev_refs.append((r["ctx"], tr, A_, B_, C_))

    log.info(f"Stage B: {len(rev_prompts)} reveal calls")
    rev_out = await run_calls(rev_prompts, strong, sem, prog, costs, "reveal")

    for (ctx, tr, A_, B_, C_), rv in zip(rev_refs, rev_out):
        k = f"combination_num_to_share={len(A_)}_num_not_to_share={len(B_)}_trial={tr}"
        b = ctx.setdefault(k, {}).setdefault("model_solution", {})
        revealed_json = parse_json(rv or "")
        b["revealed_attributes"] = {
            a: 1 if a in revealed_json else 0 for a in (A_ + B_ + C_)
        }
        b["revealed_attributes_explanation"] = revealed_json

    # save
    now = t.strftime("%Y_%B_%d_%I:%M:%S_%p")
    outdir = os.path.join(a.results_dir, a.model_name.replace("/", "_"), now)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.jsonl"), "w") as f:
        for prof in data:
            f.write(json.dumps(prof) + "\n")
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    log.info(f"Saved to {outdir}")
    os._exit(0)


def main():
    p = ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_host", type=str, default="localhost")
    p.add_argument("--model_port", type=int)
    p.add_argument("--strong_model_name", default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument("--strong_model_host", type=str)
    p.add_argument("--strong_model_port", type=int)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--data_file", required=True)
    p.add_argument("--num_profiles", type=int, required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--privacy_prompts_level", type=int, required=True)
    p.add_argument(
        "--combo_style",
        choices=["stratified", "full", "iid", "all_share", "all_not_share"],
        default="full",
    )
    p.add_argument("--num_trials", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--log_interval", type=float, default=1)
    a = p.parse_args()
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
