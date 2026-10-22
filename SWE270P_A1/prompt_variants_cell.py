# 8. More prompt variants within each technique (zero-shot, one-shot, chain-of-thought, few-shot)
# Two extra templates per technique with a different style or strategy, run on every bug once
# greedy and once sampled. All go through try_prompt: logged, tested, saved only if better.

def zs_reviewer(bug):
    return (f"Act as a senior Python code reviewer.\n"
            f"Bug report: `{bug['func_name']}` gives wrong results. Expected behaviour: {bug['requirement']}\n"
            f"```python\n{bug['func_src']}\n```\n"
            f"Find the faulty line, fix it, and change nothing else. "
            f"Reply with the full fixed function in one ```python code block.\n")

def zs_spec_first(bug):
    return (f"Specification: {bug['requirement']}\n\n"
            f"The implementation below does not meet the specification:\n"
            f"```python\n{bug['func_src']}\n```\n"
            f"Rewrite `{bug['func_name']}` so it meets the specification. Keep the same parameters.\n"
            f"Answer with code only, in a ```python block.\n")

def os_off_by_one(bug):
    return (f"You repair Python bugs with minimal edits.\n\n"
            f"Example\nBuggy:\n```python\ndef last_index(items):\n    return len(items)\n```\n"
            f"Bug: an index must be one less than the length.\n"
            f"Fixed:\n```python\ndef last_index(items):\n    return len(items) - 1\n```\n\n"
            f"Your turn\nBuggy:\n```python\n{bug['func_src']}\n```\n"
            f"Bug: {bug['hint']}\n"
            f"Fixed:\n")

def os_diff_style(bug):
    return (f"Example of a one-line repair:\n"
            f"- old: `if n % 2 == 1:   # should detect even numbers`\n"
            f"+ new: `if n % 2 == 0:`\n\n"
            f"Repair this function the same way, changing as few lines as possible:\n"
            f"```python\n{bug['func_src']}\n```\n"
            f"Requirement: {bug['requirement']}\n"
            f"Return the complete repaired function in a ```python code block.\n")

def cot_trace_test(bug):
    return (f"You are debugging this function:\n```python\n{bug['func_src']}\n```\n"
            f"These tests fail:\n{_baseline_failures(bug)}\n\n"
            f"Step 1: pick one failing test and trace what the code does on it.\n"
            f"Step 2: name the line where the result goes wrong.\n"
            f"Step 3: write the corrected `{bug['func_name']}` in a ```python code block.\n")

def cot_questions(bug):
    return (f"Answer these questions briefly, then fix the code.\n"
            f"1. What should `{bug['func_name']}` do? ({bug['requirement']})\n"
            f"2. Which line does something different from that?\n"
            f"3. What is the smallest change that fixes it?\n\n"
            f"```python\n{bug['func_src']}\n```\n"
            f"Finish with the fixed function in a ```python code block.\n")

def fs_algorithms(bug):
    return (f"Examples of fixed algorithm bugs:\n\n"
            f"Buggy: `low = mid`  →  Fixed: `low = mid + 1`  (binary search must move past mid)\n"
            f"Buggy: `visited.add(start)` missing  →  Fixed: add `visited.add(node)` before exploring "
            f"(graph search must mark nodes)\n"
            f"Buggy: `return a - b`  →  Fixed: `return abs(a - b)`  (a distance is never negative)\n\n"
            f"Now fix:\n```python\n{bug['func_src']}\n```\n"
            f"Requirement: {bug['requirement']}\n"
            f"Output only the fixed function in a ```python code block.\n")

def fs_qa(bug):
    return (f"Q: `def area(w, h): return w + h` should return the area. Fix it.\n"
            f"A:\n```python\ndef area(w, h):\n    return w * h\n```\n"
            f"Q: `def is_empty(xs): return len(xs) > 0` should say whether xs is empty. Fix it.\n"
            f"A:\n```python\ndef is_empty(xs):\n    return len(xs) == 0\n```\n"
            f"Q: The function below should satisfy: {bug['requirement']} Fix it.\n"
            f"```python\n{bug['func_src']}\n```\n"
            f"A:\n")

VARIANTS = [
    ("zero-shot", zs_reviewer), ("zero-shot", zs_spec_first),
    ("one-shot", os_off_by_one), ("one-shot", os_diff_style),
    ("chain-of-thought", cot_trace_test), ("chain-of-thought", cot_questions),
    ("few-shot", fs_algorithms), ("few-shot", fs_qa),
]

for bug_id in TARGET_FUNCTION:
    bug = load_bug(bug_id)
    for technique, build in VARIANTS:
        for do_sample in (False, True):
            try_prompt(bug_id, technique, build(bug), do_sample=do_sample)
    print("-" * 80)
