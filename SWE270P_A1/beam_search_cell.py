# 9. Wider search per line: beam search + large sampling batches, with the response prefix
# Gemma's single most likely answer is often wrong, but the right line can still be among its
# top guesses. For each faulty line this asks for the 20 best beam-search continuations plus
# 60 sampled ones (both prompt styles), and the unit tests choose. Only unsolved bugs run.
import importlib, itertools
from collections import Counter

N_BEAMS = 20
N_SAMPLES = 60
CANDIDATES_PER_LINE = 25   # distinct candidates kept per faulty line
COMBO_LIMIT = 20000        # multi-line bugs: candidate combinations screened (fast, ~10 ms each)


def generate_candidates(prompt, prefix, max_new_tokens=30):
    """Beam-search and sampled continuations of ```python + prefix, as full first lines."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text + "```python\n" + prefix, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    n_prompt = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        beams = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                               num_beams=N_BEAMS, num_return_sequences=N_BEAMS)
        samples = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=1.2, top_p=0.98, top_k=100,
                                 num_return_sequences=N_SAMPLES)
    lines = []
    for ids in list(beams) + list(samples):
        cont = tokenizer.decode(ids[n_prompt:], skip_special_tokens=True)
        lines.append((prefix + cont).split("\n")[0].replace("```", "").strip())
    return lines[:N_BEAMS], lines[N_BEAMS:]


import multiprocessing as mp

def _child_run(work_dir, module_name):
    """Runs in a forked process: import the test module fresh, run its tests in file order."""
    sys.dont_write_bytecode = True      # buggy.py changes every candidate; never reuse a .pyc
    sys.path.insert(0, str(work_dir))
    passed = 0
    try:
        mod = importlib.import_module(module_name)
        for name, fn in list(vars(mod).items()):
            if name.startswith("test") and callable(fn):
                try:
                    fn()
                    passed += 1
                except BaseException:
                    pass
    except BaseException:
        pass
    os._exit(passed)

def _fast_count(bug_id, module, timeout=5):
    p = mp.get_context("fork").Process(target=_child_run, args=(WORK_DIR.resolve(), f"{bug_id}.{module}"))
    p.start()
    p.join(timeout)
    if p.is_alive():                    # infinite loop
        p.kill()
        p.join()
        return 0
    return max(p.exitcode or 0, 0)

def _fast_screen(bug_id, source):
    """Quick (official, extra) pass counts used only to rank candidates; saving re-checks with pytest."""
    (WORK_DIR / bug_id / "buggy.py").write_text(source, encoding="utf-8")
    official = _fast_count(bug_id, "test_bug")
    extra = _fast_count(bug_id, "test_extra") if official == 3 else 0
    return official, extra


def _ensure_baseline(bug_id, bug):
    """With no patch yet, the original buggy.py is the score to beat (and keeps the file present)."""
    if current_best(bug_id) is None:
        BEST[bug_id] = {"technique": "original buggy.py (no improvement yet)", "output": None, "code": None,
                        "merged": bug["source"], "result": run_tests(bug_id, bug["source"])}
        save_final_patch(f"{bug_id}.py", f"```python\n{bug['source']}```")


def wide_search_repair(bug_id):
    if is_solved(current_best(bug_id)):
        print(f"{bug_id} | already solved ({current_best(bug_id)['technique']}), skipping")
        return
    bug = load_bug(bug_id)
    _ensure_baseline(bug_id, bug)
    tasks = LINE_TASKS[bug_id]
    options = []
    for kind, line, problem in tasks:
        prefix = PREFIXES[(bug_id, line)]
        ranked = Counter()
        for style, prompt in build_line_prompts(bug, kind, line, problem):
            beams, samples = generate_candidates(prompt, prefix)
            log_prompt("line-level + prefill (beam search + sampling)",
                       f"{prompt}\n[response prefix]\n```python\n{prefix}",
                       completion=json.dumps({"beams": beams, "samples": samples}),
                       bug_file=f"{bug_id}.py", model=MODEL_ID)
            for rank, c in enumerate(beams):
                ranked[c] += N_BEAMS - rank        # better beams count more
            for c in samples:
                ranked[c] += 1
        ranked.pop(line.strip(), None)
        ranked.pop("", None)
        top = [c for c, _ in ranked.most_common(CANDIDATES_PER_LINE)]
        print(f"{bug_id} | {kind} `{line}` -> {len(top)} candidates, e.g. {top[:4]}")
        # last option: leave this line unchanged, so partial fixes of multi-line bugs are kept
        options.append([(kind, line, c) for c in top] + [("replace", line, line.strip())])

    run_tests(bug_id, bug["source"])   # sets up work/<bug> with test_bug.py and test_extra.py
    combos = sorted(itertools.product(*[list(enumerate(o)) for o in options]),
                    key=lambda combo: sum(rank for rank, _ in combo))[:COMBO_LIMIT]
    best_quick = (-1, -1)
    for combo in combos:
        source = _apply_line_fixes(bug["source"], [fix for _, fix in combo])
        if source is None:
            continue
        quick = _fast_screen(bug_id, source)
        if quick <= best_quick:
            continue
        best_quick = quick
        attempt = {"technique": "line-level + prefill (beam search + sampling)", "output": None,
                   "code": None, "merged": source, "result": run_tests(bug_id, source)}
        if _score(attempt) > _score(current_best(bug_id)):
            BEST[bug_id] = attempt
            save_final_patch(f"{bug_id}.py", f"```python\n{source}```")
        if is_solved(attempt):
            break
    print(f"{bug_id} | best: {_fmt(current_best(bug_id)['result'])}"
          f"{'  SOLVED' if is_solved(current_best(bug_id)) else ''}")


for bug_id in TARGET_FUNCTION:
    wide_search_repair(bug_id)
    print("-" * 80)
