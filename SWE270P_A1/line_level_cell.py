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


for bug_id in TARGET_FUNCTION:
    line_level_repair(bug_id)
    print("-" * 80)
