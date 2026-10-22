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


for bug_id in TARGET_FUNCTION:
    prefill_repair(bug_id)
    print("-" * 80)
