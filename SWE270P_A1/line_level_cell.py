# 6. Line-level (fault-localized) repair
# Gemma 270M mostly copies a whole buggy function back unchanged. Here each prompt shows only the
# suspicious line plus a description of the problem, and asks for just that line rewritten.
# The fixed lines are put back into the original buggy.py and tested like every other attempt.
import itertools

LINE_SAMPLES = 5   # per line: 1 greedy + 4 sampled answers
MAX_COMBOS = 64    # bugs with several lines: test combinations of the candidates

LINE_FEW_SHOT = """You are a helpful coding assistant. You fix one line of Python at a time.

Example 1
Buggy line:
```python
if x < best:
```
Problem: the function should keep the largest value, so the comparison is reversed.
Fixed line:
```python
if x > best:
```

Example 2
Buggy line:
```python
total = p + p
```
Problem: the running total must add p to total, not p to itself.
Fixed line:
```python
total = total + p
```

Example 3
Buggy line:
```python
out.append(n)
```
Problem: n is never decreased. Rewrite the line followed by the missing line that lowers n by 1.
Fixed line:
```python
out.append(n)
n -= 1
```
"""

# (exact buggy line, description of the problem) for each bug
LINE_TASKS = {
    "BUG000": [("distance = elem - elem2",
                "the distance between two numbers must never be negative; use the absolute value of the difference.")],
    "BUG001": [("get(unvisited_nodes, nextnode) + length_by_edge[node, nextnode]",
                "the new distance to nextnode must be `distance` (the distance of the current node) plus the "
                "edge length, not get(unvisited_nodes, nextnode) plus the edge length.")],
    "BUG002": [("weight_by_edge[u, v] = min(",
                "the relaxed weight must be stored in weight_by_node[v]; the input dict weight_by_edge must not change.")],
    "BUG003": [("group_by_node[node].update(group_by_node[u])",
                "every node of the merged group must point to the same set object as u: assign "
                "group_by_node[u] to group_by_node[node] instead of calling update.")],
    "BUG004": [("length_by_path[i, k] + length_by_path[j, k]",
                "a path from i to j through k is (i to k) + (k to j), so the second term must be length_by_path[k, j].")],
    "BUG005": [("queue.append(goalnode)",
                "breadth-first search must start from startnode, so startnode is the first node put in the queue."),
               ("while True:",
                "the loop must stop when the queue is empty, so it should loop only while queue is not empty.")],
    "BUG006": [("else:",
                "in the else branch the current node must first be added to the set nodesvisited. "
                "Rewrite the line followed by the missing line."),
               ("search_from(nextnode) for nextnode in node",
                "the search must iterate over node.successors, not over node itself.")],
    "BUG007": [("node.successor = prevnode",
                "after reversing the pointer, prevnode must be set to the current node. "
                "Rewrite the line followed by the missing line.")],
    "BUG008": [("if hare is None:",
                "the hare moves two steps, so return False when hare is None or hare.successor is None."),
               ("hare = hare.successor",
                "the hare must move two steps per iteration, i.e. hare.successor.successor.")],
    "BUG009": [("dp[i] = (dp[i - 2] if i >= 2 else 0) + 1",
                "a matched pair '()' has length 2, so add 2 instead of 1."),
               ('elif i - dp[i - 1] > 0 or s[i - dp[i - 1] - 1] == "(":',
                "both conditions must hold (the index is valid AND the character there is '('), "
                "so use `and` instead of `or`."),
               ("+ 1",
                "this term counts the matched pair, which has length 2, so it must be + 2.")],
}


def build_line_prompt(buggy_line, problem):
    return (f"{LINE_FEW_SHOT}\n"
            f"Now fix this line.\n"
            f"Buggy line:\n```python\n{buggy_line}\n```\n"
            f"Problem: {problem}\n"
            f"Output only the fixed line in a ```python code block.\n")


def _parse_line_fix(output, buggy_line):
    """The rewritten line(s) from a model answer, or None if it gave nothing new."""
    blocks = _CODE_BLOCK.findall(output)
    text = textwrap.dedent(blocks[0] if blocks else output)
    lines = [l.rstrip() for l in text.splitlines()
             if l.strip() and not re.match(r"\s*(```|#|Fixed|Buggy|Problem|Here|Output)", l)]
    lines = lines[:2]
    if not lines or (len(lines) == 1 and lines[0].strip() == buggy_line.strip()):
        return None
    return tuple(lines)


def _apply_line_fixes(source, fixes):
    """Replace each buggy line (matched exactly once, ignoring indentation) with its fix."""
    lines = source.splitlines()
    for buggy, new in fixes:
        hits = [i for i, l in enumerate(lines) if l.strip() == buggy.strip()]
        if len(hits) != 1:
            return None
        i = hits[0]
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        base = len(new[0]) - len(new[0].lstrip())
        out = [indent + new[0].lstrip()]
        for extra in new[1:]:
            rel = max(len(extra) - len(extra.lstrip()) - base, 0)
            if new[0].rstrip().endswith(":") and rel == 0:
                rel = 4  # body of an `else:` / `if ...:` the model left unindented
            out.append(indent + " " * rel + extra.lstrip())
        lines[i:i + 1] = out
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
    options = []
    for buggy, problem in LINE_TASKS[bug_id]:
        prompt = build_line_prompt(buggy, problem)
        candidates = []
        for k in range(LINE_SAMPLES):
            out = generate_text(prompt, max_new_tokens=80, do_sample=k > 0)
            log_prompt("line-level few-shot", prompt, completion=out, bug_file=f"{bug_id}.py", model=MODEL_ID)
            fix = _parse_line_fix(out, buggy)
            if fix and fix not in candidates:
                candidates.append(fix)
        print(f"{bug_id} | line `{buggy}` -> {[' / '.join(c) for c in candidates] or 'no new candidates'}")
        options.append([(buggy, c) for c in candidates] or [(buggy, (buggy,))])

    for combo in itertools.islice(itertools.product(*options), MAX_COMBOS):
        source = _apply_line_fixes(bug["source"], combo)
        if source is None:
            continue
        attempt = {"technique": "line-level few-shot", "output": None, "code": None,
                   "merged": source, "result": run_tests(bug_id, source)}
        if _score(attempt) > _score(current_best(bug_id)):
            BEST[bug_id] = attempt
            save_final_patch(f"{bug_id}.py", f"```python\n{source}```")
        if is_solved(attempt):
            break
    print(f"{bug_id} | best: {_fmt(current_best(bug_id)['result'])}"
          f"{'  SOLVED' if is_solved(current_best(bug_id)) else ''}")


for bug_id in TARGET_FUNCTION:
    line_level_repair(bug_id)
    print("-" * 80)
