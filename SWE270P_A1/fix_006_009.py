# FOCUSED FIX for BUG006 and BUG009 (after a Colab restart)
# Before running: log in to Hugging Face, and upload to /content (the Files panel top level):
#   - the course ZIP (with data/BUG000 ... BUG009)
#   - your submission ZIP (the one with results/prompts.jsonl)
# Then run:  %run -i /content/fix_006_009.py
# It restores your submission, adds a test for BUG006's real flaw, searches only BUG006 and
# BUG009 with Gemma, and rebuilds SWE270P_A1.zip (lowercase .zip so Gradescope unzips it).

# ONE-CELL RESUME for after a Colab restart.
# Paste this whole file into a single cell and run it. It re-creates everything the notebook
# set up (model, data folder, helpers, tests, prompt builders, line-level definitions) without
# re-running the long repair loops, then runs the beam-search repair on every unsolved bug.
# Bugs already solved in results/final_patches are detected and skipped.
# If the model download fails with a 401 / "gated" error, run the login() cell first.


# ======== model + generate_text ========
# >>> MODEL START
# Load model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if "model" not in globals():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

def generate_text(prompt: str, max_new_tokens: int = 200, do_sample: bool = False,
                  temperature: float = 0.7, top_p: float = 0.95) -> str:
    """Send `prompt` as one user turn to gemma-3-270m-it and return only the newly generated text.

    Decoding is greedy by default so a prompt always gives the same answer; pass do_sample=True
    to draw a different candidate when retrying the same prompt.
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=64)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the tokens after the prompt; skip_special_tokens drops the trailing <end_of_turn>.
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# >>> MODEL END

# ======== locate data ========
import os, glob, zipfile
from pathlib import Path

# Colab's Files panel can upload to /content (the default) or, if you were browsing "..", to /.
SEARCH_DIRS = [Path("."), Path("/content"), Path("/")]

def _find_assignment_root():
    for base in SEARCH_DIRS:
        for cand in (base, base / "SWE270P_A1"):
            if (cand / "data" / "BUG000").is_dir():
                return cand.resolve()
    # Not unzipped yet: find the uploaded course ZIP and extract it (into /content on Colab).
    for base in SEARCH_DIRS:
        for z in sorted(base.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                hits = [n for n in zf.namelist()
                        if n.endswith("data/BUG000/buggy.py") and "__MACOSX" not in n]
                if hits:
                    target = Path("/content") if Path("/content").is_dir() else z.parent
                    zf.extractall(target)
                    print(f"Extracted {z} into {target}")
                    return (target / hits[0][:-len("data/BUG000/buggy.py")]).resolve()
    raise FileNotFoundError(
        "Could not find data/BUG000. Upload the course ZIP (with data/BUG000 ... data/BUG009) "
        "in the Colab Files panel, then re-run this cell."
    )

os.chdir(_find_assignment_root())
print("Working directory:", os.getcwd())
print("Bugs found:", sorted(p.name for p in Path("data").glob("BUG*")))

# ======== helpers ========
from pathlib import Path
import json, re, ast, textwrap, datetime as dt
from typing import Optional

"""
Helper function to save the prompts.jsonl file.
Helper functions to extract the repaired code from the LLM response and save the complete file.
"""

PROMPTS_PATH = Path("results/prompts.jsonl")
PATCH_DIR = Path("results/final_patches")                        # Use this folder to save the repaired Python code
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
MODEL_ID = "google/gemma-3-270m-it"

# The function each bug's test_bug.py imports, i.e. the one the model has to repair.
TARGET_FUNCTION = {
    "BUG000": "has_close_elements",
    "BUG001": "shortest_path_length",
    "BUG002": "shortest_paths",
    "BUG003": "minimum_spanning_tree",
    "BUG004": "shortest_path_lengths",
    "BUG005": "breadth_first_search",
    "BUG006": "depth_first_search",
    "BUG007": "reverse_linked_list",
    "BUG008": "detect_cycle",
    "BUG009": "longestValidParentheses",
}

def log_prompt(technique: str, prompt: str,
               completion: Optional[str] = None,
               bug_file: Optional[str] = None,
               model: Optional[str] = None,
               ):
    rec = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "technique": technique,
        "prompt": prompt,
    }
    if completion is not None: rec["completion"] = completion
    if bug_file is not None:   rec["bug_file"] = bug_file
    if model is not None:      rec["model"] = model
    with PROMPTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# A fenced block; an unterminated one (generation hit max_new_tokens) runs to the end of the text.
_CODE_BLOCK = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)

def extract_code(model_output: str, func_name: Optional[str] = None) -> Optional[str]:
    """Pull the Python code out of a model response.

    Prefers the last fenced block that defines `func_name` (in chain-of-thought answers the
    final block is the fix), then the longest fenced block, then everything from the first `def`.
    """
    blocks = [textwrap.dedent(b).strip() for b in _CODE_BLOCK.findall(model_output)]
    blocks = [b for b in blocks if b]
    if func_name:
        defining = [b for b in blocks if re.search(rf"^\s*def\s+{func_name}\s*\(", b, re.M)]
        if defining:
            return defining[-1]
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"^def\s+\w+\s*\(", model_output, re.M)
    return model_output[m.start():].strip() if m else None


def _parse_lenient(code: str):
    """ast.parse, dropping trailing lines (e.g. prose after an unfenced function) until it parses."""
    for _ in range(30):
        try:
            return code, ast.parse(code)
        except SyntaxError as e:
            if not e.lineno or e.lineno <= 1:
                return None
            code = "\n".join(code.splitlines()[:e.lineno - 1])
    return None


def _function_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def merge_into_original(original_source: str, model_code: str, func_name: str) -> Optional[str]:
    """Replace only `func_name` in the original buggy.py with the model's version.

    Imports, helper functions (e.g. BUG001's get / insert_or_update) and the module docstring are
    kept from the original, so the result is a complete drop-in replacement for buggy.py.
    Returns None if the model's code has no usable function or changes the parameters.
    """
    parsed = _parse_lenient(model_code)
    if parsed is None:
        return None
    code, tree = parsed
    new_fn = _function_node(tree, func_name)
    if new_fn is None:
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1:
            return None
        new_fn = funcs[0]  # the model renamed the function; the name is restored below

    old_fn = _function_node(ast.parse(original_source), func_name)
    if [a.arg for a in new_fn.args.args] != [a.arg for a in old_fn.args.args]:
        return None  # the autograder calls the original signature

    new_lines = code.splitlines()[new_fn.lineno - 1:new_fn.end_lineno]
    new_lines[0] = re.sub(r"def\s+\w+", f"def {func_name}", new_lines[0], count=1)
    orig_lines = original_source.splitlines()
    merged = "\n".join(orig_lines[:old_fn.lineno - 1] + new_lines + orig_lines[old_fn.end_lineno:]) + "\n"
    try:
        ast.parse(merged)
    except SyntaxError:
        return None
    return merged


def save_final_patch(bug_file: str, model_output: str) -> str:
    """bug_file is a file name like 'BUG001.py'. Saves to results/final_patches/BUG001.py

    Save the complete file (imports and helper functions included), not only the fixed function.
    """
    bug_id = Path(bug_file).stem
    func_name = TARGET_FUNCTION.get(bug_id)
    code = extract_code(model_output, func_name)
    if code is None:
        raise ValueError(f"No Python code found in the model output for {bug_file}")

    original_path = DATA_DIR / bug_id / "buggy.py"
    if func_name and original_path.exists():
        merged = merge_into_original(original_path.read_text(encoding="utf-8"), code, func_name)
        if merged is None:
            raise ValueError(f"The model output for {bug_file} has no usable `{func_name}` function")
        code = merged

    out_path = PATCH_DIR / bug_file
    out_path.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    return str(out_path)

# ======== repair toolkit ========
import os, shutil, subprocess, sys

WORK_DIR = Path("work")   # scratch copies for testing; data/ is never modified
TEST_TIMEOUT = 30         # seconds per pytest run; buggy loops can spin forever

REQUIREMENTS = {
    "BUG000": "Return True if any two different elements of `numbers` are closer to each other than "
              "`threshold` (their absolute difference is less than threshold), otherwise False.",
    "BUG001": "Dijkstra's algorithm: return the length of the shortest path from startnode to goalnode. "
              "The tentative distance of nextnode is the smaller of its current distance and "
              "(distance of the current node + length of edge (node, nextnode)). "
              "Return 0 when startnode is goalnode and float('inf') when goalnode is unreachable.",
    "BUG002": "Bellman-Ford: return a dict mapping every node to the minimum total weight of a path from "
              "source. Each round relaxes every edge (u, v) by lowering the weight of node v. "
              "Negative edges are allowed. Do not modify the input dict weight_by_edge.",
    "BUG003": "Kruskal's algorithm: return the set of edges of a minimum spanning tree. When an edge joins "
              "two components, every node of the merged component must refer to the same merged group.",
    "BUG004": "Floyd-Warshall: return a dict with the shortest path length for every ordered pair (i, j) "
              "of nodes 0..n-1. A path from i to j through k has length (i to k) + (k to j).",
    "BUG005": "Breadth-first search: return True if goalnode can be reached from startnode by following "
              "node.successors, otherwise False. The search starts at startnode and returns False "
              "once the queue is empty.",
    "BUG006": "Depth-first search: return True if goalnode can be reached from startnode by following "
              "node.successors, otherwise False. Each node is visited at most once so cycles terminate.",
    "BUG007": "Reverse a singly linked list (nodes linked by .successor) in place and return the new head. "
              "Return None for an empty list.",
    "BUG008": "Tortoise-and-hare cycle detection: return True if the linked list starting at node "
              "(linked by .successor) has a cycle, otherwise False. The tortoise moves 1 step and the "
              "hare moves 2 steps per iteration; return False as soon as the hare reaches the end.",
    "BUG009": "Return the length of the longest valid (well-formed) parentheses substring of s. "
              "Each matched pair '()' adds 2 to the length, e.g. '()' -> 2, '(())' -> 4, ')()())' -> 4.",
}

HINTS = {
    "BUG000": "The distance between two numbers must never be negative: compare the absolute difference with threshold.",
    "BUG001": "Look at the second argument of min(): the new distance to nextnode must start from `distance` "
              "(the distance of the node just popped), not from get(unvisited_nodes, nextnode).",
    "BUG002": "The relaxation result is stored in the wrong dictionary: it must update weight_by_node[v], "
              "not weight_by_edge[u, v].",
    "BUG003": "In the inner loop, `group_by_node[node].update(...)` is wrong: every node of the merged group "
              "must point to the same set object, i.e. group_by_node[node] = group_by_node[u].",
    "BUG004": "Check the indices of the second term inside min(...): the path from k to j is length_by_path[k, j].",
    "BUG005": "Two problems: the queue starts with the wrong node (it must start with startnode), "
              "and `while True` must instead loop only while the queue is not empty.",
    "BUG006": "Two problems: the node is never added to nodesvisited, and the generator iterates over "
              "`node` instead of `node.successors`.",
    "BUG007": "Inside the loop prevnode is never updated: set prevnode = node before moving to nextnode.",
    "BUG008": "The hare must move two steps (hare = hare.successor.successor), so check that both hare and "
              "hare.successor are not None before moving it; loop with `while True`.",
    "BUG009": "Each matched pair adds 2, not 1 (in both branches), and the elif condition must use `and` "
              "instead of `or`: the index must be valid AND the character there must be '('.",
}

# Our own robustness tests (the assignment asks for these). They run next to test_bug.py.
EXTRA_TESTS = {
    "BUG000": """from .buggy import has_close_elements

def test_extra_descending_values():
    # a signed difference would be negative here and wrongly count as "close"
    assert has_close_elements([5.0, 1.0, 9.0], 1.0) == False

def test_extra_close_pair():
    assert has_close_elements([1.0, 2.0, 1.5], 0.6) == True

def test_extra_empty_and_single():
    assert has_close_elements([], 1.0) == False
    assert has_close_elements([3.0], 1.0) == False
""",
    "BUG001": """from .node import Node
from .buggy import shortest_path_length

def test_extra_cheaper_two_hop_path():
    c = Node("c", None, [])
    b = Node("b", None, [c])
    a = Node("a", None, [b, c])
    assert shortest_path_length({(a, b): 1, (b, c): 1, (a, c): 5}, a, c) == 2

def test_extra_unreachable():
    a = Node("a", None, [])
    b = Node("b", None, [])
    assert shortest_path_length({}, a, b) == float("inf")
""",
    "BUG002": """from .buggy import shortest_paths

def test_extra_negative_edge_and_input_untouched():
    graph = {("A", "B"): 5, ("B", "C"): -3, ("A", "C"): 4}
    snapshot = dict(graph)
    assert shortest_paths("A", graph) == {"A": 0, "B": 5, "C": 2}
    assert graph == snapshot

def test_extra_two_hop_shortcut():
    graph = {("A", "B"): 1, ("B", "C"): 1, ("A", "C"): 5}
    assert shortest_paths("A", graph) == {"A": 0, "B": 1, "C": 2}
""",
    "BUG003": """from .buggy import minimum_spanning_tree

def test_extra_triangle():
    assert minimum_spanning_tree({(1, 2): 1, (2, 3): 2, (1, 3): 3}) == {(1, 2), (2, 3)}

def test_extra_merging_components():
    result = minimum_spanning_tree({(1, 2): 1, (3, 4): 1, (4, 5): 1, (2, 3): 2, (1, 4): 3, (1, 5): 5})
    assert result == {(1, 2), (3, 4), (4, 5), (2, 3)}
""",
    "BUG004": """from .buggy import shortest_path_lengths

def test_extra_shortcut_through_middle_node():
    result = shortest_path_lengths(3, {(0, 1): 1, (1, 2): 1, (0, 2): 5})
    assert result[0, 2] == 2
    assert result[0, 1] == 1 and result[1, 2] == 1
    assert result[2, 0] == float("inf")
    assert all(result[i, i] == 0 for i in range(3))
""",
    "BUG005": """from .node import Node
from .buggy import breadth_first_search

def test_extra_start_is_goal():
    a = Node("a", None, [])
    assert breadth_first_search(a, a)

def test_extra_direction_matters():
    c = Node("c", None, [])
    b = Node("b", None, [c])
    a = Node("a", None, [b])
    assert breadth_first_search(a, c)
    assert not breadth_first_search(c, a)

def test_extra_cycle_without_goal_terminates():
    a = Node("a", None, [])
    b = Node("b", None, [a])
    a.successors = [b]
    assert not breadth_first_search(a, Node("c", None, []))
""",
    "BUG006": """from .node import Node
from .buggy import depth_first_search

def test_extra_start_is_goal():
    a = Node("a", None, [])
    assert depth_first_search(a, a)

def test_extra_direction_matters():
    c = Node("c", None, [])
    b = Node("b", None, [c])
    a = Node("a", None, [b])
    assert depth_first_search(a, c)
    assert not depth_first_search(c, a)

def test_extra_cycle_without_goal_terminates():
    a = Node("a", None, [])
    b = Node("b", None, [a])
    a.successors = [b]
    assert not depth_first_search(a, Node("c", None, []))
""",
    "BUG007": """from .node import Node
from .buggy import reverse_linked_list

def test_extra_two_nodes():
    tail = Node("b")
    head = Node("a", tail)
    assert reverse_linked_list(head) is tail
    assert tail.successor is head
    assert head.successor is None
""",
    "BUG008": """from .node import Node
from .buggy import detect_cycle

def _chain(n):
    nodes = [Node(i) for i in range(n)]
    for a, b in zip(nodes, nodes[1:]):
        a.successor = b
    return nodes

def test_extra_single_node():
    assert not detect_cycle(Node(1))

def test_extra_self_loop():
    n = Node(1)
    n.successor = n
    assert detect_cycle(n)

def test_extra_even_length_without_cycle():
    assert not detect_cycle(_chain(4)[0])

def test_extra_cycle_in_tail():
    nodes = _chain(4)
    nodes[3].successor = nodes[1]
    assert detect_cycle(nodes[0])
""",
    "BUG009": """from .buggy import longestValidParentheses

def test_extra_empty():
    assert longestValidParentheses("") == 0

def test_extra_no_valid_pair():
    assert longestValidParentheses(")(") == 0

def test_extra_unclosed_prefix():
    assert longestValidParentheses("(()") == 2

def test_extra_leading_close():
    assert longestValidParentheses(")()())") == 4

def test_extra_adjacent_and_nested():
    assert longestValidParentheses("()(())") == 6
""",
}


def load_bug(bug_id: str) -> dict:
    source = (DATA_DIR / bug_id / "buggy.py").read_text(encoding="utf-8")
    func_name = TARGET_FUNCTION[bug_id]
    tree = ast.parse(source)
    fn = _function_node(tree, func_name)
    helpers = [ast.get_source_segment(source, n) for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name != func_name]
    return {
        "bug_id": bug_id,
        "func_name": func_name,
        "source": source,
        "func_src": "\n".join(source.splitlines()[fn.lineno - 1:fn.end_lineno]),
        "helpers": helpers,
        "tests": (DATA_DIR / bug_id / "test_bug.py").read_text(encoding="utf-8"),
        "requirement": REQUIREMENTS[bug_id],
        "hint": HINTS[bug_id],
    }


def _pytest(test_file: Path) -> dict:
    total = len(re.findall(r"^def test", test_file.read_text(), re.M))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line", str(test_file)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TEST_TIMEOUT,
                              env={**os.environ, "COLUMNS": "200"})  # don't truncate FAILED lines
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": total,
                "log": f"TIMEOUT: {test_file.name} did not finish in {TEST_TIMEOUT}s (infinite loop?)"}
    log = re.sub(r"\S*/(test_\w+\.py)", r"\1", proc.stdout + proc.stderr)
    m = re.search(r"(\d+) passed", log)
    return {"passed": int(m.group(1)) if m else 0, "total": total, "log": log}


def run_tests(bug_id: str, source: str) -> dict:
    """Test `source` as that bug's buggy.py against test_bug.py and our extra tests."""
    dest = WORK_DIR / bug_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_DIR / bug_id, dest,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".DS_Store"))
    (dest / "buggy.py").write_text(source, encoding="utf-8")
    (dest / "test_extra.py").write_text(EXTRA_TESTS[bug_id], encoding="utf-8")
    return {"official": _pytest(dest / "test_bug.py"), "extra": _pytest(dest / "test_extra.py")}


def failure_summary(result: dict, max_lines: int = 12) -> str:
    """One line per failing test (with its assert message), error or timeout, to show the model."""
    lines = []
    for part in ("official", "extra"):
        for line in result[part]["log"].splitlines():
            if re.match(r"(FAILED|ERROR|TIMEOUT)", line) and line not in lines:
                lines.append(line.strip())
    return "\n".join(lines[:max_lines]) or "(no failure details)"


# Sanity check: every original buggy.py should fail at least one official test.
for bug_id in TARGET_FUNCTION:
    r = run_tests(bug_id, load_bug(bug_id)["source"])
    print(f"{bug_id} original: official {r['official']['passed']}/{r['official']['total']}, "
          f"extra {r['extra']['passed']}/{r['extra']['total']}")

# ======== prompt builders ========
SYSTEM_LINE = "You are a helpful coding assistant."

def _output_rule(bug):
    return (f"Output: Return only the corrected `{bug['func_name']}` function (same name and parameters) "
            "in a single ```python code block. Do not add explanations or tests.")

def _helpers_block(bug):
    if not bug["helpers"]:
        return ""
    return ("These helper functions already exist and are correct (do not change them):\n"
            "```python\n" + "\n\n".join(bug["helpers"]) + "\n```\n")

def _task_block(bug):
    return (f"Function:\n```python\n{bug['func_src']}\n```\n"
            f"{_helpers_block(bug)}"
            f"Requirement: {bug['requirement']}\n")


def build_zero_shot(bug):
    return (f"{SYSTEM_LINE}\n\n"
            f"Fix the bug in this Python function.\n"
            f"{_task_block(bug)}"
            f"{_output_rule(bug)}\n")


ONE_SHOT_EXAMPLE = """Example (Buggy → Fixed)
Problem: find_max(nums) should return the largest element, but it returns the smallest.
Buggy:
```python
def find_max(nums):
    best = nums[0]
    for x in nums:
        if x < best:
            best = x
    return best
```
Fixed:
```python
def find_max(nums):
    best = nums[0]
    for x in nums:
        if x > best:
            best = x
    return best
```
"""

def build_one_shot(bug):
    return (f"{SYSTEM_LINE}\n\n"
            f"{ONE_SHOT_EXAMPLE}\n"
            f"Now fix a new function (One-Shot):\n"
            f"{_task_block(bug)}"
            f"{_output_rule(bug)}\n")


def build_cot(bug):
    return (f"{SYSTEM_LINE}\n\n"
            f"Task: Fix the bug in this function.\n"
            f"{_task_block(bug)}\n"
            f"Think step by step in 2-4 short bullet points: what the function should do, "
            f"which line is wrong, and how to fix it.\n"
            f"After your reasoning, output ONLY the final corrected `{bug['func_name']}` function "
            f"in a ```python code block and nothing else.\n")


FEW_SHOT_EXAMPLES = """Here are three examples (Buggy → Fixed):

Problem: wrong comparison operator. find_max(nums) should return the largest element.
Buggy:
```python
def find_max(nums):
    best = nums[0]
    for x in nums:
        if x < best:
            best = x
    return best
```
Fixed:
```python
def find_max(nums):
    best = nums[0]
    for x in nums:
        if x > best:
            best = x
    return best
```

Problem: wrong variable. total(prices) should return the sum of prices.
Buggy:
```python
def total(prices):
    result = 0
    for p in prices:
        result = p + p
    return result
```
Fixed:
```python
def total(prices):
    result = 0
    for p in prices:
        result = result + p
    return result
```

Problem: missing statement. count_down(n) should return [n, n-1, ..., 1].
Buggy:
```python
def count_down(n):
    out = []
    while n > 0:
        out.append(n)
    return out
```
Fixed:
```python
def count_down(n):
    out = []
    while n > 0:
        out.append(n)
        n -= 1
    return out
```
"""

def build_few_shot(bug):
    return (f"{SYSTEM_LINE}\n\n"
            f"{FEW_SHOT_EXAMPLES}\n"
            f"Now fix a new function:\n"
            f"{_task_block(bug)}"
            f"{_output_rule(bug)}\n")


_BASELINE = {}  # bug_id -> pytest result of the original buggy.py

def _baseline_failures(bug):
    if bug["bug_id"] not in _BASELINE:
        _BASELINE[bug["bug_id"]] = run_tests(bug["bug_id"], bug["source"])
    return failure_summary(_BASELINE[bug["bug_id"]])

def build_tests_aware(bug, hint=False):
    return (f"{SYSTEM_LINE}\n\n"
            f"The function below fails its unit tests. Fix it so that all tests pass.\n"
            f"{_task_block(bug)}"
            f"Unit tests:\n```python\n{bug['tests'].strip()}\n```\n"
            f"Current pytest failures:\n{_baseline_failures(bug)}\n"
            + (f"Hint: {bug['hint']}\n" if hint else "")
            + f"{_output_rule(bug)}\n")

def build_tests_aware_hint(bug):
    return build_tests_aware(bug, hint=True)


def build_self_refine(bug, previous_code, failures):
    return (f"{SYSTEM_LINE}\n\n"
            f"You tried to fix `{bug['func_name']}` and wrote this code:\n"
            f"```python\n{previous_code}\n```\n"
            f"{_helpers_block(bug)}"
            f"It still fails these tests:\n{failures}\n"
            f"Requirement: {bug['requirement']}\n"
            f"Hint: {bug['hint']}\n"
            f"Review your code line by line, find what is still wrong, and fix it.\n"
            f"{_output_rule(bug)}\n")


print(build_zero_shot(load_bug("BUG000")))

# ======== try_prompt ========
MAX_NEW_TOKENS = 512
BEST = {}  # bug_id -> best attempt so far (the one saved in results/final_patches)

def _score(attempt):
    if not attempt or not attempt.get("result"):
        return (-1, -1)
    r = attempt["result"]
    return (r["official"]["passed"], r["extra"]["passed"])

def is_solved(attempt) -> bool:
    if not attempt or not attempt.get("result"):
        return False
    r = attempt["result"]
    return all(r[p]["passed"] == r[p]["total"] for p in ("official", "extra"))

def current_best(bug_id):
    """Best patch so far, including one saved by an earlier run of this notebook."""
    if bug_id not in BEST:
        saved = PATCH_DIR / f"{bug_id}.py"
        if saved.exists():
            source = saved.read_text(encoding="utf-8")
            BEST[bug_id] = {"technique": "(saved earlier)", "output": None, "merged": source,
                            "code": None, "result": run_tests(bug_id, source)}
    return BEST.get(bug_id)

def _fmt(result):
    if result is None:
        return "no usable code in the response"
    o, e = result["official"], result["extra"]
    return f"official {o['passed']}/{o['total']}, extra {e['passed']}/{e['total']}"

def try_prompt(bug_id: str, technique: str, prompt: str, do_sample: bool = False,
               max_new_tokens: int = MAX_NEW_TOKENS, show_output: bool = False) -> dict:
    bug = load_bug(bug_id)
    output = generate_text(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
    log_prompt(technique, prompt, completion=output, bug_file=f"{bug_id}.py", model=MODEL_ID)

    code = extract_code(output, bug["func_name"])
    merged = merge_into_original(bug["source"], code, bug["func_name"]) if code else None
    result = run_tests(bug_id, merged) if merged else None
    attempt = {"technique": technique, "output": output, "code": code, "merged": merged, "result": result}

    saved = ""
    if result is not None and _score(attempt) > _score(current_best(bug_id)):
        BEST[bug_id] = attempt
        save_final_patch(f"{bug_id}.py", output)
        saved = "  -> saved as final patch"
    print(f"{bug_id} | {technique:<18} | {'sampled' if do_sample else 'greedy ':<7} | {_fmt(result)}{saved}")
    if show_output:
        print(output[:1500])
    return attempt


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


# ======== restore the submission (log + patches) ========
import glob, zipfile
def _log_lines(z):
    with zipfile.ZipFile(z) as zf:
        return sum(1 for _ in zf.open("results/prompts.jsonl"))
_subs = []
for _base in ("/content", ".", ".."):
    for _z in glob.glob(f"{_base}/*.zip") + glob.glob(f"{_base}/*.ZIP"):
        with zipfile.ZipFile(_z) as _zf:
            if "results/prompts.jsonl" in _zf.namelist():
                _subs.append(_z)
if not _subs:
    raise FileNotFoundError("Upload your submission ZIP (the one containing results/prompts.jsonl) first.")
_sub = max(_subs, key=_log_lines)
shutil.rmtree("results", ignore_errors=True)
with zipfile.ZipFile(_sub) as _zf:
    _zf.extractall(".")
BEST.clear()
print(f"Restored {_sub}: {_log_lines(_sub)} logged prompts + 10 patches")


# ======== round-2 definitions ========
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


# ======== focus on BUG006 and BUG009 ========
# The autograder's hidden tests failed these two. BUG006's patch marks only the start node as
# visited, so a cycle that does not pass through the start recurses forever; test that too.
EXTRA_TESTS["BUG006"] += """

def test_extra_cycle_away_from_start():
    a = Node("a", None, [])
    b = Node("b", None, [a])
    a.successors = [b]
    s = Node("s", None, [a])
    assert not depth_first_search(s, Node("g", None, []))
"""

for bug_id in ("BUG006", "BUG009"):
    print(f"{bug_id} | current patch: {_fmt(current_best(bug_id)['result'])}")
    for _round in range(3):
        round2_repair(bug_id)
        if is_solved(current_best(bug_id)):
            break
    print("=" * 80)


# ======== rebuild the submission ZIP ========
expected = [f"BUG{i:03d}.py" for i in range(10)]
for bug_id in TARGET_FUNCTION:
    best = current_best(bug_id)
    print(f"{bug_id}  {_fmt(best['result']):<34}{'yes' if is_solved(best) else 'NO'}")
ZIP_NAME = "SWE270P_A1.zip"
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("results/", "")
    zf.writestr("results/final_patches/", "")
    zf.write(PROMPTS_PATH, "results/prompts.jsonl")
    for name in expected:
        zf.write(PATCH_DIR / name, f"results/final_patches/{name}")
print(f"\nWrote {Path(ZIP_NAME).resolve()} with {_log_lines(ZIP_NAME)} logged prompts")
try:
    from google.colab import files
    files.download(ZIP_NAME)
except ImportError:
    pass
