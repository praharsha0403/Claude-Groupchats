# RESUME CELL: run this after a Colab restart, once the setup cells have run again
# (login, Load model, Locate data, helpers, Repair toolkit, Prompt builders, try_prompt).
# It re-creates the line-level definitions (sections 6-7, without re-running their loops)
# and then runs the beam-search repair (section 8) on every bug that is not solved yet.
# Bugs already solved in results/final_patches are detected and skipped.

# ---- from section 6 (definitions only) ----
# 6. Line-level (fault-localized) repair with many samples, chosen by the tests
# Gemma 270M copies whatever code is in the prompt: whole functions, and even few-shot examples.
# So these prompts have no examples and ask for ONE line only: either the faulty line rewritten,
# or the single line that is missing. Each line is asked many times (two prompt styles, greedy +
# sampled); answers that are just echoes or unrelated code are dropped, the most frequent
# candidates are kept, and the unit tests decide which combination becomes the patch.
import itertools
from collections import Counter
from difflib import SequenceMatcher

LINE_SAMPLES = 10   # per prompt style and line: 1 greedy + 9 sampled answers
TOP_CANDIDATES = 6  # most frequent distinct candidates kept per line
MAX_COMBOS = 64     # bugs with several lines: candidate combinations tested

# (kind, exact line in buggy.py, problem). kind "replace": that line is wrong.
# kind "insert_after": a line is missing right after it.
LINE_TASKS = {
    "BUG000": [("replace", "distance = elem - elem2",
                "the distance between two numbers must never be negative; use the absolute value of the difference.")],
    "BUG001": [("replace", "get(unvisited_nodes, nextnode) + length_by_edge[node, nextnode]",
                "the new distance to nextnode must be `distance` (the distance of the current node) plus the "
                "edge length, not get(unvisited_nodes, nextnode) plus the edge length.")],
    "BUG002": [("replace", "weight_by_edge[u, v] = min(",
                "the relaxed weight must be stored in weight_by_node[v]; the input dict weight_by_edge must not change.")],
    "BUG003": [("replace", "group_by_node[node].update(group_by_node[u])",
                "every node of the merged group must point to the same set object as u: assign "
                "group_by_node[u] to group_by_node[node] instead of calling update.")],
    "BUG004": [("replace", "length_by_path[i, k] + length_by_path[j, k]",
                "a path from i to j through k is (i to k) + (k to j), so the second term must be length_by_path[k, j].")],
    "BUG005": [("replace", "queue.append(goalnode)",
                "breadth-first search must start from startnode, so startnode is the first node put in the queue."),
               ("replace", "while True:",
                "the loop must stop when the queue is empty, so it should loop only while queue is not empty.")],
    "BUG006": [("insert_after", "else:",
                "the current node is never recorded as visited; the missing line adds node to the set nodesvisited."),
               ("replace", "search_from(nextnode) for nextnode in node",
                "the search must iterate over node.successors, not over node itself.")],
    "BUG007": [("insert_after", "node.successor = prevnode",
                "prevnode is never updated; the missing line sets prevnode to the current node."),],
    "BUG008": [("replace", "if hare is None:",
                "the hare moves two steps, so return False when hare is None or hare.successor is None."),
               ("replace", "hare = hare.successor",
                "the hare must move two steps per iteration, i.e. hare.successor.successor.")],
    "BUG009": [("replace", "dp[i] = (dp[i - 2] if i >= 2 else 0) + 1",
                "a matched pair '()' has length 2, so add 2 instead of 1."),
               ("replace", 'elif i - dp[i - 1] > 0 or s[i - dp[i - 1] - 1] == "(":',
                "both conditions must hold (the index is valid AND the character there is '('), "
                "so use `and` instead of `or`."),
               ("replace", "+ 1",
                "this term counts the matched pair, which has length 2, so it must be + 2.")],
}


def build_line_prompts(bug, kind, line, problem):
    """Two prompt styles per line: the line alone, and the line inside its function."""
    if kind == "replace":
        alone = (f"You are a helpful coding assistant.\n\n"
                 f"This line of Python code has a bug:\n```python\n{line}\n```\n"
                 f"Problem: {problem}\n"
                 f"Write the corrected line. Output only that one line in a ```python code block.\n")
        context = (f"You are a helpful coding assistant.\n\n"
                   f"Here is a Python function:\n```python\n{bug['func_src']}\n```\n"
                   f"The line `{line}` has a bug. Problem: {problem}\n"
                   f"Write the corrected version of that one line. "
                   f"Output only that line in a ```python code block.\n")
    else:
        alone = (f"You are a helpful coding assistant.\n\n"
                 f"In a Python function, a line is missing right after this line:\n```python\n{line}\n```\n"
                 f"Problem: {problem}\n"
                 f"Write only the missing line in a ```python code block.\n")
        context = (f"You are a helpful coding assistant.\n\n"
                   f"Here is a Python function:\n```python\n{bug['func_src']}\n```\n"
                   f"A line is missing right after the line `{line}`. Problem: {problem}\n"
                   f"Write only the missing line in a ```python code block.\n")
    return [("line-level zero-shot", alone), ("line-level + context", context)]


def _answer_lines(output):
    """Every candidate line in an answer: fenced code, `inline code`, then plain text."""
    fenced = _CODE_BLOCK.findall(output)
    prose = _CODE_BLOCK.sub("\n", output)
    texts = fenced + re.findall(r"`([^`\n]+)`", prose) + [prose]
    lines = []
    for text in texts:
        for l in text.splitlines():
            l = re.sub(r"^\s*(?:(?:Fixed|Corrected|Missing|New)\s+line\s*:\s*)", "", l).strip()
            if l and "`" not in l and not l.startswith("#") and l not in lines:
                lines.append(l)
    return lines


def _candidates_from(output, kind, line, func_lines):
    if kind == "replace":
        return [l for l in _answer_lines(output)
                if l != line and SequenceMatcher(None, l, line).ratio() >= 0.5]
    keep = []
    for l in _answer_lines(output):
        if l in func_lines or len(l) > 100 or l.startswith(("def ", "class ", "import ", "from ", "return ")):
            continue
        try:
            ast.parse(l)
        except SyntaxError:
            continue
        keep.append(l)
    return keep


def _apply_line_fixes(source, fixes):
    """fixes: (kind, line, new_text). Each anchor line must match exactly once (ignoring indentation)."""
    lines = source.splitlines()
    for kind, line, new in fixes:
        hits = [i for i, l in enumerate(lines) if l.strip() == line.strip()]
        if len(hits) != 1:
            return None
        i = hits[0]
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        if kind == "replace":
            lines[i] = indent + new
        else:
            body = "    " if lines[i].rstrip().endswith(":") else ""
            lines.insert(i + 1, indent + body + new)
    patched = "\n".join(lines) + "\n"
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return patched


def line_level_repair(bug_id):
    if not FORCE and is_solved(current_best(bug_id)):
        print(f"{bug_id} | already solved ({current_best(bug_id)['technique']}), skipping")
        return
    bug = load_bug(bug_id)
    func_lines = {l.strip() for l in bug["func_src"].splitlines()}
    options = []
    for kind, line, problem in LINE_TASKS[bug_id]:
        votes = Counter()
        for technique, prompt in build_line_prompts(bug, kind, line, problem):
            for k in range(LINE_SAMPLES):
                out = generate_text(prompt, max_new_tokens=60, do_sample=k > 0, temperature=1.0)
                log_prompt(technique, prompt, completion=out, bug_file=f"{bug_id}.py", model=MODEL_ID)
                votes.update(set(_candidates_from(out, kind, line, func_lines)))
        top = [c for c, _ in votes.most_common(TOP_CANDIDATES)]
        print(f"{bug_id} | {kind} `{line}` -> {top or 'no candidates'}")
        options.append([(kind, line, c) for c in top] or [("replace", line, line)])

    # Try the combinations with the most-voted candidates first.
    combos = sorted(itertools.product(*[list(enumerate(o)) for o in options]),
                    key=lambda combo: sum(rank for rank, _ in combo))[:MAX_COMBOS]
    for combo in combos:
        source = _apply_line_fixes(bug["source"], [fix for _, fix in combo])
        if source is None:
            continue
        attempt = {"technique": "line-level", "output": None, "code": None,
                   "merged": source, "result": run_tests(bug_id, source)}
        if _score(attempt) > _score(current_best(bug_id)):
            BEST[bug_id] = attempt
            save_final_patch(f"{bug_id}.py", f"```python\n{source}```")
        if is_solved(attempt):
            break
    print(f"{bug_id} | best: {_fmt(current_best(bug_id)['result'])}"
          f"{'  SOLVED' if is_solved(current_best(bug_id)) else ''}")


# ---- from section 7 (definitions only) ----
# 7. Line-level repair with a response prefix ("prefill")
# Gemma 270M often echoes the line or rewrites a different one. Here the start of its answer is
# written for it (the response prefix below), so it only generates the rest of the faulty line.
# Uses LINE_TASKS / build_line_prompts / _apply_line_fixes from section 6; only unsolved bugs run.
import itertools
from collections import Counter

PREFILL_SAMPLES = 20   # per line: 1 greedy + 19 sampled continuations

# Start of the answer for each (bug, line). The model writes everything after it.
PREFIXES = {
    ("BUG000", "distance = elem - elem2"): "distance = ",
    ("BUG001", "get(unvisited_nodes, nextnode) + length_by_edge[node, nextnode]"): "",
    ("BUG002", "weight_by_edge[u, v] = min("): "weight_by_",
    ("BUG003", "group_by_node[node].update(group_by_node[u])"): "group_by_node[node]",
    ("BUG004", "length_by_path[i, k] + length_by_path[j, k]"): "length_by_path[i, k] + length_by_path[",
    ("BUG005", "queue.append(goalnode)"): "queue.append(",
    ("BUG005", "while True:"): "while ",
    ("BUG006", "else:"): "nodesvisited.",
    ("BUG006", "search_from(nextnode) for nextnode in node"): "search_from(nextnode) for nextnode in node",
    ("BUG007", "node.successor = prevnode"): "prevnode = ",
    ("BUG008", "if hare is None:"): "if hare is None or ",
    ("BUG008", "hare = hare.successor"): "hare = hare.successor",
    ("BUG009", "dp[i] = (dp[i - 2] if i >= 2 else 0) + 1"): "dp[i] = (dp[i - 2] if i >= 2 else 0) + ",
    ("BUG009", 'elif i - dp[i - 1] > 0 or s[i - dp[i - 1] - 1] == "(":'): "elif i - dp[i - 1] > 0 ",
    ("BUG009", "+ 1"): "+ ",
}


def generate_with_prefix(prompt, prefix, max_new_tokens=40, do_sample=False, temperature=1.0):
    """Like generate_text, but the model's answer starts with ```python + `prefix`."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text + "```python\n" + prefix, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=0.95, top_k=64)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return prefix + tokenizer.decode(new_tokens, skip_special_tokens=True)


def prefill_repair(bug_id):
    if is_solved(current_best(bug_id)):
        print(f"{bug_id} | already solved ({current_best(bug_id)['technique']}), skipping")
        return
    bug = load_bug(bug_id)
    options = []
    for kind, line, problem in LINE_TASKS[bug_id]:
        prefix = PREFIXES[(bug_id, line)]
        _, prompt = build_line_prompts(bug, kind, line, problem)[1]   # the "+ context" style
        logged_prompt = f"{prompt}\n[response prefix]\n```python\n{prefix}"
        votes = Counter()
        for k in range(PREFILL_SAMPLES):
            out = generate_with_prefix(prompt, prefix, do_sample=k > 0)
            log_prompt("line-level + prefill", logged_prompt, completion=out,
                       bug_file=f"{bug_id}.py", model=MODEL_ID)
            first = out.split("\n")[0].replace("```", "").strip()
            if first and first != line.strip():
                votes[first] += 1
        top = [c for c, _ in votes.most_common(TOP_CANDIDATES)]
        print(f"{bug_id} | {kind} `{line}` -> {top or 'no candidates'}")
        options.append([(kind, line, c) for c in top] or [("replace", line, line)])

    combos = sorted(itertools.product(*[list(enumerate(o)) for o in options]),
                    key=lambda combo: sum(rank for rank, _ in combo))[:MAX_COMBOS]
    for combo in combos:
        source = _apply_line_fixes(bug["source"], [fix for _, fix in combo])
        if source is None:
            continue
        attempt = {"technique": "line-level + prefill", "output": None, "code": None,
                   "merged": source, "result": run_tests(bug_id, source)}
        if _score(attempt) > _score(current_best(bug_id)):
            BEST[bug_id] = attempt
            save_final_patch(f"{bug_id}.py", f"```python\n{source}```")
        if is_solved(attempt):
            break
    print(f"{bug_id} | best: {_fmt(current_best(bug_id)['result'])}"
          f"{'  SOLVED' if is_solved(current_best(bug_id)) else ''}")


# ---- section 8: beam search + sampling ----
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


def wide_search_repair(bug_id):
    if is_solved(current_best(bug_id)):
        print(f"{bug_id} | already solved ({current_best(bug_id)['technique']}), skipping")
        return
    bug = load_bug(bug_id)
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
        options.append([(kind, line, c) for c in top] or [("replace", line, line)])

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
