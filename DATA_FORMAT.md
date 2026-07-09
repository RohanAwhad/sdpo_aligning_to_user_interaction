# SDFT/SDPO Training Data Format

## File format
JSONL — one JSON object per line.

## Schema

```jsonl
{
  "prompt": [
    {"from": "human", "value": "<user question/prompt>"}
  ],
  "completion": {
    "from": "gpt",
    "value": "<model's response to the prompt>"
  },
  "user_response": {
    "from": "human",
    "value": "<hindsight signal>"
  }
}
```

## Fields

### `prompt` (required)
List of message dicts. Single-turn: one entry with `"from": "human"`.
Alternative key format: `{"role": "user", "content": "..."}` (both supported, collator normalizes).

### `completion` (required for off-policy, ignored for on-policy)
The model's response that will be scored under both base and hindsight contexts.
- **Off-policy**: read from the dataset (pre-generated)
- **On-policy** (`--on_policy`): ignored — generated from the current model at each step

### `user_response` (required)
The hindsight signal `o` appended to the prompt to form the conditional context `(x, o)`.

For **SDPO** (paper): the user's follow-up message after seeing the completion.
For **SDFT** (our adaptation): the golden/reference answer.

Injected into the prompt as:
```
The following is the correct answer. Use this to guide your response: {o}
```

## Example

```json
{
  "prompt": [
    {
      "from": "human",
      "value": "How does MaaS handle rate limiting across tiers?"
    }
  ],
  "completion": {
    "from": "gpt",
    "value": "MaaS enforces rate limits per tier using policy checks..."
  },
  "user_response": {
    "from": "human",
    "value": "Each tier defines its own rate-limit and quota. When a request arrives, the controller assigns it to a tier based on group membership and checks the tier's limits before forwarding."
  }
}
```

## Notes
- `completion` values should NOT contain thinking tags — use `enable_thinking=False` when generating
- `user_response` is used as-is for the hindsight context; quality of this field directly affects training signal
- For multi-turn prompts, all messages go in the `prompt` list; the collator normalizes `from/value` to `role/content`
