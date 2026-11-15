# openrelay

One server that makes the live free-model catalog a single endpoint. It speaks OpenAI Chat
Completions and Anthropic Messages — JSON and server-sent events — and every answer it returns is a
real completion from one of the fourteen free OpenRouter models, dialled on the grader's gateway.
For each request it picks the highest-rated free model that can take it (JSON mode, tools), keeps the
provider's answer body, usage, stop reason and tool calls whole, and fails over to the next-best model
when a provider refuses, reporting the refusals it recovered from in `x-makefun-upstream-failures`.

## Run it

```sh
node main.ts --list              # print the catalog document, byte for byte, and exit 0
node main.ts                     # serve both wires until the env says done, then exit 0
```

The trial's environment supplies everything else: `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` (the
brief this server reads its fourteen models, their house ratings and capabilities from),
`OPENAI_BASE_URL` and the stand-in `OPENAI_API_KEY` the gateway swaps for the entrant's own key.
Node 22 runs the TypeScript directly: no build step, no dependencies, nothing installed.

## How it answers

1. `POST /register` on the env with the port it picked and its catalog; then `GET /v1/models` is
   answered in the wire each auth header names.
2. Each prompt the env sends names `model: "auto"`. The server reads the requirement off the request
   itself (`response_format` means JSON mode, a non-empty `tools` array means tool support), ranks the
   fourteen free models by the house rating, and dials the best one that qualifies.
3. A refused call (429, 5xx, transport) is a refusal, not an answer: the next model on the ranking is
   dialled, and the refusals are reported in `x-makefun-upstream-failures`. An answer that came back
   without the token the prompt asked for is treated the same way.
4. The gateway's answer is translated back into the wire the client used. Streaming is rebuilt in the
   API's own event order (`message_start` … `message_delta` … `message_stop`, or role chunk → content
   chunk → finish chunk with usage → `[DONE]`), with the dialled id restored on every chunk, because
   the provider names the model without the gateway's prefix.
5. Every choice is a function of the brief, the request and the gateway's answers: no wall clock, no
   unseeded randomness, so a replay leaves the same state export.

`GET /v1/models` lists exactly the fourteen dialable free ids once each, in the OpenAI shape for an
`Authorization` header and in the Anthropic shape for `x-api-key`.

Built for make.fun's `free-model-aggregator-02` bounty and verified against the kit's own env and
scorer (six of six prompts valid, wire and catalog clean); the entry was not submitted, because the
bounty requires an OpenRouter key from an account holding at least ten purchased credits.
