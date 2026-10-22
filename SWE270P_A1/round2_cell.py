# 10. Round 2: much wider sampling, building on partial fixes
# Run after resume_all_in_one.py (it uses its definitions). For each unsolved bug:
#  - start from the best patch so far when it beats the original (e.g. keep lines already fixed),
#    and only search the faulty lines that are still unchanged
#  - per line: 20 beams + 400 samples at two temperatures, both prompt styles, with the prefix
#  - screen candidate combinations with the fast tester, confirm improvements with pytest
import itertools
from collections import Counter

ROUND2_SAMPLES = 200            # per temperature and prompt style
ROUND2_TEMPERATURES = (0.8, 1.5)
ROUND2_PER_LINE = 40


def sample_candidates(prompt, prefix, max_new_tokens=30):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text + "```python\n" + prefix, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]
    seqs = []
    with torch.inference_mode():
        seqs += list(model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    num_beams=N_BEAMS, num_return_sequences=N_BEAMS))
        for temp in ROUND2_TEMPERATURES:
            for _ in range(ROUND2_SAMPLES // 100):
                seqs += list(model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                            temperature=temp, top_p=0.99, top_k=0,
                                            num_return_sequences=100))
    lines = [(prefix + tokenizer.decode(s[n_prompt:], skip_special_tokens=True)).split("\n")[0]
             .replace("```", "").strip() for s in seqs]
    return lines[:N_BEAMS], lines[N_BEAMS:]


def _base_source(bug_id, bug):
    """The best patch so far if it passes more official tests than the original, else the original."""
    best = current_best(bug_id)
    original = run_tests(bug_id, bug["source"])["official"]["passed"]
    if best and best.get("merged") and best["result"]["official"]["passed"] > original:
        return best["merged"]
    return bug["source"]


def _ensure_baseline(bug_id, bug):
    """With no patch yet, the original buggy.py is the score to beat (and keeps the file present)."""
    if current_best(bug_id) is None:
        BEST[bug_id] = {"technique": "original buggy.py (no improvement yet)", "output": None, "code": None,
                        "merged": bug["source"], "result": run_tests(bug_id, bug["source"])}
        save_final_patch(f"{bug_id}.py", f"```python\n{bug['source']}```")


def round2_repair(bug_id):
    if is_solved(current_best(bug_id)):
        print(f"{bug_id} | already solved ({current_best(bug_id)['technique']}), skipping")
        return
    bug = load_bug(bug_id)
    _ensure_baseline(bug_id, bug)
    base = _base_source(bug_id, bug)
    base_lines = {l.strip() for l in base.splitlines()}
    open_tasks = [t for t in LINE_TASKS[bug_id] if t[1].strip() in base_lines]
    if base != bug["source"] and not open_tasks:
        # every faulty line was already changed, yet the patch still fails: search all lines again
        base, open_tasks = bug["source"], list(LINE_TASKS[bug_id])
        print(f"{bug_id} | best patch changed every faulty line but still fails; searching all "
              f"{len(open_tasks)} line(s) again from the original")
    elif base != bug["source"]:
        print(f"{bug_id} | building on best patch so far; still searching {len(open_tasks)} line(s)")
    options = []
    for kind, line, problem in open_tasks:
        prefix = PREFIXES[(bug_id, line)]
        ranked = Counter()
        for style, prompt in build_line_prompts(bug, kind, line, problem):
            beams, samples = sample_candidates(prompt, prefix)
            log_prompt("line-level + prefill (round 2: wide sampling)",
                       f"{prompt}\n[response prefix]\n```python\n{prefix}",
                       completion=json.dumps({"beams": beams, "samples": samples[:400]}),
                       bug_file=f"{bug_id}.py", model=MODEL_ID)
            for rank, c in enumerate(beams):
                ranked[c] += N_BEAMS - rank
            for c in samples:
                ranked[c] += 1
        ranked.pop(line.strip(), None)
        ranked.pop("", None)
        top = [c for c, _ in ranked.most_common(ROUND2_PER_LINE)]
        print(f"{bug_id} | {kind} `{line}` -> {len(top)} candidates, e.g. {top[:5]}")
        # last option: leave this line unchanged, so partial fixes of multi-line bugs are kept
        options.append([(kind, line, c) for c in top] + [("replace", line, line.strip())])

    if not any(len(o) > 1 for o in options):
        print(f"{bug_id} | no new candidates")
        return
    run_tests(bug_id, base)   # sets up work/<bug>
    combos = sorted(itertools.product(*[list(enumerate(o)) for o in options]),
                    key=lambda combo: sum(rank for rank, _ in combo))[:COMBO_LIMIT]
    best_quick = (-1, -1)
    for combo in combos:
        source = _apply_line_fixes(base, [fix for _, fix in combo])
        if source is None:
            continue
        quick = _fast_screen(bug_id, source)
        if quick <= best_quick:
            continue
        best_quick = quick
        attempt = {"technique": "line-level + prefill (round 2: wide sampling)", "output": None,
                   "code": None, "merged": source, "result": run_tests(bug_id, source)}
        if _score(attempt) > _score(current_best(bug_id)):
            BEST[bug_id] = attempt
            save_final_patch(f"{bug_id}.py", f"```python\n{source}```")
        if is_solved(attempt):
            break
    print(f"{bug_id} | best: {_fmt(current_best(bug_id)['result'])}"
          f"{'  SOLVED' if is_solved(current_best(bug_id)) else ''}")


for bug_id in TARGET_FUNCTION:
    round2_repair(bug_id)
    print("-" * 80)
