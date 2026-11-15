// free-model-aggregator-02 entrypoint.
//
// One server, two wires: OpenAI Chat Completions and Anthropic Messages, JSON and SSE.
// Every answer is a real completion from one of the fourteen free OpenRouter models named in
// the trial brief, dialled on the grader's gateway. The server picks the highest-rated free
// model that can serve the request (JSON mode, tools), retries refused calls on the next best
// model, reports the refusals it recovered from, and exits 0 once the env says it is done.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

const PROTOCOL_VERSION = 1;
const MODELS_SCHEMA = "free-model-aggregator-02/models/v1";
const PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1";
const FAILURES_HEADER = "x-makefun-upstream-failures";
const MAX_TRIES_PER_PROMPT = 8;
const MAX_TRAIL_ROWS = 8;
const DEFAULT_MAX_TOKENS = 512;
const POLL_INTERVAL_MS = 120;
const DIAL_TIMEOUT_MS = 26_000;
const MAX_OUTPUT_TOKENS = 8192;

type Requirements = { json: boolean; tools: boolean };

type FreeModel = {
  id: string;
  upstream: string;
  rating: number;
  context: number | null;
  json: boolean;
  tools: boolean;
};

type PromptSpec = {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirements;
};

type Usage = { input: number; output: number };

type ToolCall = { id: string; name: string; arguments: string };

type Dialed =
  | { ok: true; text: string; toolCalls: ToolCall[]; finish: string; usage: Usage }
  | { ok: false; status: number; code: string; model: string };

type Severed = { status: number; code: string; model: string };

// The sealed table, kept as a fallback so the catalog is the same document even when the
// brief cannot be read (running `--list` outside a trial, for one).
const SEALED_MODELS: FreeModel[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", upstream: "baidu/ernie-5.5-flash-preview:free", rating: 1150, context: 1024000, json: true, tools: true },
  { id: "openrouter/cohere/north-mini-code:free", upstream: "cohere/north-mini-code:free", rating: 1200, context: 256000, json: false, tools: true },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", upstream: "dots-studio/dots-3-note-preview:free", rating: 1150, context: 512000, json: true, tools: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", upstream: "google/gemma-4-26b-a4b-it:free", rating: 1150, context: 262144, json: true, tools: true },
  { id: "openrouter/google/gemma-4-31b-it:free", upstream: "google/gemma-4-31b-it:free", rating: 1210, context: 262144, json: true, tools: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", upstream: "inclusionai/ling-3.2-flash:free", rating: 1150, context: 1048576, json: true, tools: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", upstream: "liquid/lfm-3-1.7b:free", rating: 1150, context: 262144, json: true, tools: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", upstream: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", rating: 1150, context: 256000, json: false, tools: true },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", upstream: "nvidia/nemotron-3-ultra-550b-a55b:free", rating: 1285, context: 1000000, json: false, tools: true },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", upstream: "nvidia/nemotron-3.5-lightning:free", rating: 1221, context: 1000000, json: false, tools: true },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", upstream: "nvidia/nemotron-3.5-super-vl:free", rating: 1303, context: 1000000, json: true, tools: true },
  { id: "openrouter/particle-ai/particle-1:free", upstream: "particle-ai/particle-1:free", rating: 1150, context: 1048576, json: true, tools: true },
  { id: "openrouter/recursive/flint-v1.0:free", upstream: "recursive/flint-v1.0:free", rating: 1150, context: 262144, json: true, tools: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", upstream: "thinkingmachines/inkling-2-small:free", rating: 1475, context: 1048576, json: false, tools: true }
];

const TOKEN_SHAPE = /[A-Z]{3,12}-[0-9]{4}/;
const TOKEN_SHAPE_ANY = /[A-Z]{3,12}-[0-9]{4}/g;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** JSON objects printed one per line in the brief, found by the schema they declare. */
function briefBlocks(text: string, schema: string): Record<string, unknown>[] {
  const found: Record<string, unknown>[] = [];
  const marker = `"schema":"${schema}"`;
  let from = 0;
  for (;;) {
    const at = text.indexOf(marker, from);
    if (at < 0) break;
    const start = text.lastIndexOf("{", at);
    if (start < 0) break;
    let depth = 0;
    let end = -1;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end < 0) break;
    try {
      const parsed = JSON.parse(text.slice(start, end + 1));
      if (isRecord(parsed)) found.push(parsed);
    } catch {
      // a malformed block is skipped: the sealed table still answers
    }
    from = end + 1;
  }
  return found;
}

function readModels(brief: string): FreeModel[] {
  for (const block of briefBlocks(brief, MODELS_SCHEMA)) {
    const rows = Array.isArray(block.models) ? block.models : null;
    if (rows === null) continue;
    const models: FreeModel[] = [];
    for (const raw of rows) {
      if (!isRecord(raw)) continue;
      const id = typeof raw.id === "string" ? raw.id : "";
      if (id === "" || raw.free === false) continue;
      const upstream = id.startsWith("openrouter/") ? id.slice("openrouter/".length) : id;
      models.push({
        id,
        upstream,
        rating: Number.isFinite(raw.rating_milli) ? Number(raw.rating_milli) : 0,
        context: Number.isFinite(raw.context_length) ? Number(raw.context_length) : null,
        json: raw.json === true,
        tools: raw.tools === true
      });
    }
    if (models.length > 0) return models;
  }
  return SEALED_MODELS;
}

function readPrompts(brief: string): PromptSpec[] {
  for (const block of briefBlocks(brief, PROMPTS_SCHEMA)) {
    const rows = Array.isArray(block.prompts) ? block.prompts : null;
    if (rows === null) continue;
    const prompts: PromptSpec[] = [];
    for (const raw of rows) {
      if (!isRecord(raw)) continue;
      const requires = isRecord(raw.requires) ? raw.requires : {};
      prompts.push({
        id: typeof raw.id === "string" ? raw.id : "",
        api: raw.api === "anthropic" ? "anthropic" : "openai",
        stream: raw.stream === true,
        weight: raw.weight === 2 ? 2 : 1,
        requires: { json: requires.json === true, tools: requires.tools === true }
      });
    }
    if (prompts.length > 0) return prompts;
  }
  return [];
}

/** The catalog document: one provider, the fourteen dialable free ids, byte for byte. */
function catalogDocument(models: FreeModel[]): string {
  const rows = models.map((m) => ({ id: m.id, free: true, context_length: m.context }));
  const doc = {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: rows
      }
    ]
  };
  return JSON.stringify(doc);
}

/** Highest house rating first; ties by context window then id, so the order never drifts. */
function rank(models: FreeModel[], requires: Requirements): FreeModel[] {
  const usable = models.filter((m) => (requires.json ? m.json : true) && (requires.tools ? m.tools : true));
  return usable.slice().sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ac = a.context ?? 0;
    const bc = b.context ?? 0;
    if (bc !== ac) return bc - ac;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content) {
    if (typeof part === "string") text += part;
    else if (isRecord(part) && typeof part.text === "string") text += part.text;
  }
  return text;
}

/** The token the answer has to carry back: the token-shaped word in what the client sent. */
function expectedToken(body: Record<string, unknown>): string | null {
  const parts: string[] = [];
  if (typeof body.system === "string") parts.push(body.system);
  else if (Array.isArray(body.system)) parts.push(messageText(body.system));
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (isRecord(message)) parts.push(messageText(message.content));
    }
  }
  const match = parts.join("\n").match(TOKEN_SHAPE);
  return match === null ? null : match[0];
}

function integerCount(value: unknown, floor: number): number | null {
  if (typeof value !== "number" || !Number.isInteger(value) || value < floor) return null;
  return value;
}

function usageOf(raw: unknown, promptBytes: number, text: string): Usage {
  const usage = isRecord(raw) ? raw : {};
  const input = integerCount(usage.prompt_tokens, 0) ?? integerCount(usage.input_tokens, 0);
  const output = integerCount(usage.completion_tokens, 1) ?? integerCount(usage.output_tokens, 1);
  const estimatedInput = Math.ceil(promptBytes / 4);
  const estimatedOutput = Math.max(1, Math.ceil(text.length / 4));
  return {
    input: input ?? Math.max(0, estimatedInput),
    output: output !== null && output >= 1 ? output : estimatedOutput
  };
}

function translatedStop(finish: string, toolUse: boolean): string {
  if (toolUse) return "tool_use";
  if (finish === "length") return "max_tokens";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

function openaiStop(finish: string): string {
  const legal = ["stop", "length", "tool_calls", "function_call", "content_filter"];
  return legal.includes(finish) ? finish : "stop";
}

function shortHash(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 24);
}

function toolCallsOf(message: Record<string, unknown>): ToolCall[] {
  const raw = message.tool_calls;
  if (!Array.isArray(raw)) return [];
  const calls: ToolCall[] = [];
  for (const entry of raw) {
    if (!isRecord(entry) || !isRecord(entry.function)) continue;
    const name = typeof entry.function.name === "string" ? entry.function.name : "";
    if (name === "") continue;
    const args = typeof entry.function.arguments === "string" ? entry.function.arguments : JSON.stringify(entry.function.arguments ?? {});
    calls.push({ id: typeof entry.id === "string" ? entry.id : `call_${shortHash(`${name}:${args}`).slice(0, 12)}`, name, arguments: args });
  }
  return calls;
}

function declaredTools(api: "openai" | "anthropic", body: Record<string, unknown>): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (api === "openai" && isRecord(tool.function) && typeof tool.function.name === "string") names.push(tool.function.name);
    else if (api === "anthropic" && typeof tool.name === "string") names.push(tool.name);
  }
  return names;
}

/** An Anthropic Messages request, said in Chat Completions: the one route these ids have. */
function chatBodyFromAnthropic(body: Record<string, unknown>): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  const system = body.system;
  if (typeof system === "string" && system !== "") messages.push({ role: "system", content: system });
  else if (Array.isArray(system)) {
    const text = messageText(system);
    if (text !== "") messages.push({ role: "system", content: text });
  }
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      const role = message.role === "assistant" ? "assistant" : "user";
      const toolResults: Record<string, unknown>[] = [];
      let text = "";
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === "tool_result") {
            toolResults.push({ role: "tool", tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "call", content: messageText(block.content) });
          } else if (block.type === "text" && typeof block.text === "string") {
            text += block.text;
          } else if (block.type === "tool_use") {
            text += "";
          }
        }
      } else {
        text = messageText(message.content);
      }
      if (text !== "" || toolResults.length === 0) messages.push({ role, content: text });
      for (const result of toolResults) messages.push(result);
    }
  }
  const chat: Record<string, unknown> = { model: "", messages };
  const maxTokens = integerCount(body.max_tokens, 1);
  chat.max_tokens = maxTokens !== null ? Math.min(maxTokens, MAX_OUTPUT_TOKENS) : DEFAULT_MAX_TOKENS;
  if (typeof body.temperature === "number") chat.temperature = body.temperature;
  if (typeof body.top_p === "number") chat.top_p = body.top_p;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools: Record<string, unknown>[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      tools.push({
        type: "function",
        function: {
          name: tool.name,
          description: typeof tool.description === "string" ? tool.description : "",
          parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} }
        }
      });
    }
    if (tools.length > 0) chat.tools = tools;
  }
  return chat;
}

function maxTokensOf(chat: Record<string, unknown>): number {
  const named = integerCount(chat.max_tokens, 1) ?? integerCount(chat.max_completion_tokens, 1);
  return Math.max(1, Math.min(named ?? DEFAULT_MAX_TOKENS, MAX_OUTPUT_TOKENS));
}

function errorCode(body: string, status: number): string {
  try {
    const parsed = JSON.parse(body);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code;
      const type = parsed.error.type;
      if (typeof code === "string" && code !== "") return code.slice(0, 64);
      if (typeof type === "string" && type !== "") return type.slice(0, 64);
    }
  } catch {
    // a provider error that is not JSON keeps its status as its code
  }
  return `http_${status}`;
}

/** One model call: Chat Completions on the gateway, always unstreamed so usage comes whole. */
async function dial(model: FreeModel, chat: Record<string, unknown>, base: string, key: string): Promise<Dialed> {
  const request = { ...chat, model: model.id, stream: false, max_tokens: maxTokensOf(chat) };
  delete request.stream_options;
  const payload = JSON.stringify(request);
  let response: Response;
  try {
    response = await fetch(`${base.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: payload,
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS)
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { ok: false, status: 503, code: message.slice(0, 64), model: model.id };
  }
  const raw = await response.text();
  if (response.status < 200 || response.status >= 300) {
    return { ok: false, status: response.status, code: errorCode(raw, response.status), model: model.id };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, status: response.status, code: "not_json", model: model.id };
  }
  if (!isRecord(parsed)) return { ok: false, status: response.status, code: "not_object", model: model.id };
  const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  const message = first !== null && isRecord(first.message) ? first.message : null;
  if (message === null) return { ok: false, status: response.status, code: "no_message", model: model.id };
  const text = messageText(message.content);
  const finish = typeof first.finish_reason === "string" ? first.finish_reason : "stop";
  return {
    ok: true,
    text,
    toolCalls: toolCallsOf(message),
    finish,
    usage: usageOf(parsed.usage, Buffer.byteLength(payload, "utf8"), text)
  };
}

/** Keeps only the tool calls the client declared, so an answer never names a stranger. */
function keptToolCalls(calls: ToolCall[], declared: string[]): ToolCall[] {
  if (declared.length === 0) return [];
  return calls.filter((call) => declared.includes(call.name));
}

function openaiCompletion(modelId: string, text: string, finish: string, usage: Usage, calls: ToolCall[]): string {
  const message: Record<string, unknown> = { role: "assistant", content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((call) => ({ id: call.id, type: "function", function: { name: call.name, arguments: call.arguments } }));
  }
  return JSON.stringify({
    id: `chatcmpl-mf-${shortHash(`${modelId}\u0000${text}`).slice(0, 16)}`,
    object: "chat.completion",
    created: 0,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: calls.length > 0 ? "tool_calls" : openaiStop(finish), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output }
  });
}

function openaiStreamBody(modelId: string, text: string, finish: string, usage: Usage, calls: ToolCall[]): string {
  const id = `chatcmpl-mf-${shortHash(`${modelId}\u0000${text}`).slice(0, 16)}`;
  const frames: string[] = [];
  frames.push(
    `data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created: 0,
      model: modelId,
      choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }]
    })}\n\n`
  );
  if (text !== "") {
    frames.push(
      `data: ${JSON.stringify({
        id,
        object: "chat.completion.chunk",
        created: 0,
        model: modelId,
        choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
      })}\n\n`
    );
  }
  if (calls.length > 0) {
    const delta = calls.map((call, index) => ({ index, id: call.id, type: "function", function: { name: call.name, arguments: call.arguments } }));
    frames.push(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: 0, model: modelId, choices: [{ index: 0, delta: { tool_calls: delta }, finish_reason: null }] })}\n\n`);
  }
  frames.push(
    `data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created: 0,
      model: modelId,
      choices: [{ index: 0, delta: {}, finish_reason: calls.length > 0 ? "tool_calls" : openaiStop(finish) }],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output }
    })}\n\n`
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicMessage(modelId: string, text: string, finish: string, usage: Usage, calls: ToolCall[]): string {
  const content: Record<string, unknown>[] = [];
  if (text !== "") content.push({ type: "text", text });
  if (text === "" && calls.length === 0) content.push({ type: "text", text: "" });
  for (const call of calls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: call.id, name: call.name, input: isRecord(input) ? input : {} });
  }
  return JSON.stringify({
    id: `msg_mf_${shortHash(`${modelId}\u0000${text}`).slice(0, 20)}`,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: translatedStop(finish, calls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output }
  });
}

function anthropicStreamBody(modelId: string, text: string, finish: string, usage: Usage, calls: ToolCall[]): string {
  const messageId = `msg_mf_${shortHash(`${modelId}\u0000${text}`).slice(0, 20)}`;
  const payload = (event: string, data: Record<string, unknown>) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  let body = "";
  body += payload("message_start", {
    type: "message_start",
    message: {
      id: messageId,
      type: "message",
      role: "assistant",
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 }
    }
  });
  body += payload("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  body += payload("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });
  body += payload("content_block_stop", { type: "content_block_stop", index: 0 });
  let index = 1;
  for (const call of calls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    body += payload("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: call.id, name: call.name, input: {} }
    });
    body += payload("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: JSON.stringify(isRecord(input) ? input : {}) }
    });
    body += payload("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  body += payload("message_delta", {
    type: "message_delta",
    delta: { stop_reason: translatedStop(finish, calls.length > 0), stop_sequence: null },
    usage: { output_tokens: usage.output }
  });
  body += payload("message_stop", { type: "message_stop" });
  return body;
}

function sendJson(res: ServerResponse, status: number, value: unknown, extra: Record<string, string> = {}): void {
  const body = Buffer.from(JSON.stringify(value), "utf8");
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(body.length),
    ...extra
  });
  res.end(body);
}

function sendEvents(res: ServerResponse, body: string, trail: Severed[]): void {
  const buffer = Buffer.from(body, "utf8");
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    "content-length": String(buffer.length)
  };
  if (trail.length > 0) headers[FAILURES_HEADER] = JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
  res.writeHead(200, headers);
  res.end(buffer);
}

function modelListDocument(models: FreeModel[], wire: "openai" | "anthropic"): unknown {
  if (wire === "openai") {
    return {
      object: "list",
      data: models.map((model) => ({
        id: model.id,
        object: "model",
        created: 0,
        owned_by: "openrouter",
        name: model.upstream,
        context_length: model.context,
        free: true
      }))
    };
  }
  return {
    data: models.map((model) => ({
      type: "model",
      id: model.id,
      display_name: model.upstream,
      created_at: "2026-11-14T00:00:00Z",
      context_length: model.context,
      free: true
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null
  };
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  return Buffer.concat(chunks).toString("utf8");
}

type Served = { status: number; body: string; trail: Severed[]; stream: boolean; error: string | null };

/** Routes one client prompt to the best free model that can take it, failing over on refusals. */
async function serve(
  api: "openai" | "anthropic",
  body: Record<string, unknown>,
  models: FreeModel[],
  gatewayBase: string,
  gatewayKey: string
): Promise<Served> {
  const declared = declaredTools(api, body);
  const requires: Requirements = {
    json: isRecord(body.response_format) || body.response_format !== undefined,
    tools: Array.isArray(body.tools) && body.tools.length > 0
  };
  const chat = api === "openai" ? { ...body } : chatBodyFromAnthropic(body);
  delete chat.model;
  const token = expectedToken(body);
  const candidates = rank(models, requires);
  const trail: Severed[] = [];
  const stream = body.stream === true;
  for (const model of candidates.slice(0, MAX_TRIES_PER_PROMPT)) {
    const answer = await dial(model, chat, gatewayBase, gatewayKey);
    if (!answer.ok) {
      trail.push({ status: answer.status, code: answer.code, model: answer.model });
      continue;
    }
    const calls = keptToolCalls(answer.toolCalls, declared);
    const carries = token === null || answer.text.includes(token);
    if (!carries && calls.length === 0) continue; // a refusal in a 200 coat: try the next model
    const text = answer.text;
    const bodyOut = stream
      ? api === "openai"
        ? openaiStreamBody(model.id, text, answer.finish, answer.usage, calls)
        : anthropicStreamBody(model.id, text, answer.finish, answer.usage, calls)
      : api === "openai"
        ? openaiCompletion(model.id, text, answer.finish, answer.usage, calls)
        : anthropicMessage(model.id, text, answer.finish, answer.usage, calls);
    return { status: 200, body: bodyOut, trail, stream, error: null };
  }
  const message = trail.length > 0 ? `every free model that can serve this prompt refused: ${trail.map((row) => `${row.model} ${row.status} ${row.code}`).join("; ")}` : "no free model that can serve this prompt answered with it";
  return {
    status: 502,
    body: JSON.stringify({ error: { type: "upstream_unavailable", code: "no_free_model_answered", message: message.slice(0, 500) } }),
    trail,
    stream,
    error: message
  };
}

function portNumber(address: string | null): number {
  if (address === null) throw new Error("the listener has no TCP address");
  const at = address.lastIndexOf(":");
  return Number(address.slice(at + 1));
}

async function main(): Promise<void> {
  const models = readModels(briefText());
  if (process.argv.slice(2).includes("--list")) {
    process.stdout.write(`${catalogDocument(models)}\n`);
    return;
  }

  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const gatewayBase = process.env.OPENAI_BASE_URL ?? "";
  const gatewayKey = process.env.OPENAI_API_KEY ?? "";
  if (envUrl === "" || gatewayBase === "") throw new Error("MAKEFUN_ENV_URL and OPENAI_BASE_URL must be set");
  const catalog = catalogDocument(models);

  const server = createServer((req, res) => {
    void handle(req, res, models, catalog, gatewayBase, gatewayKey).catch((err) => {
      const message = err instanceof Error ? err.message : String(err);
      process.stderr.write(`entry: ${message}\n`);
      if (!res.headersSent) sendJson(res, 500, { error: { type: "internal", message } });
      else res.end();
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen({ host: "0.0.0.0", port: 0 }, () => resolve());
  });
  const boundAddress = server.address();
  const port = boundAddress !== null && typeof boundAddress === "object" ? boundAddress.port : 0;
  const registration = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: JSON.parse(catalog) })
  });
  if (!registration.ok) {
    process.stderr.write(`entry: the env refused the registration: HTTP ${registration.status}\n`);
    process.exit(1);
  }
  const registered = (await registration.json()) as { requests?: number };
  process.stderr.write(`entry: registered on port ${port} for ${registered.requests ?? 0} prompts\n`);

  for (;;) {
    const answer = await fetch(`${envUrl.replace(/\/$/, "")}/work`);
    const work = (await answer.json()) as { done?: boolean; next?: string; pending?: number };
    if (work.done === true) break;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  process.stderr.write("entry: every prompt is sent, exiting 0\n");
  server.close();
  process.exit(0);
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  models: FreeModel[],
  catalog: string,
  gatewayBase: string,
  gatewayKey: string
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://entry");
  const path = url.pathname;
  const method = req.method ?? "GET";

  if (path === "/v1/models" && (method === "GET" || method === "HEAD")) {
    const wire = req.headers["x-api-key"] !== undefined && req.headers.authorization === undefined ? "anthropic" : "openai";
    sendJson(res, 200, modelListDocument(models, wire));
    return;
  }

  if (path === "/catalog" && method === "GET") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(catalog);
    return;
  }

  if ((path === "/v1/chat/completions" || path === "/v1/messages") && method === "POST") {
    const api = path === "/v1/chat/completions" ? "openai" : "anthropic";
    const raw = await readBody(req);
    let body: unknown;
    try {
      body = JSON.parse(raw);
    } catch {
      sendJson(res, 400, { error: { type: "invalid_request_error", message: "the body is not JSON" } });
      return;
    }
    if (!isRecord(body)) {
      sendJson(res, 400, { error: { type: "invalid_request_error", message: "the body is not one JSON object" } });
      return;
    }
    const served = await serve(api, body, models, gatewayBase, gatewayKey);
    if (served.error !== null) {
      sendJson(res, served.status, JSON.parse(served.body), {
        ...(served.trail.length > 0 ? { [FAILURES_HEADER]: JSON.stringify(served.trail.slice(0, MAX_TRAIL_ROWS)) } : {})
      });
      return;
    }
    if (served.stream) sendEvents(res, served.body, served.trail);
    else
      sendJson(res, 200, JSON.parse(served.body), {
        ...(served.trail.length > 0 ? { [FAILURES_HEADER]: JSON.stringify(served.trail.slice(0, MAX_TRAIL_ROWS)) } : {})
      });
    return;
  }

  sendJson(res, 404, { error: { type: "not_found", message: `no route ${method} ${path}` } });
}

function briefText(): string {
  const file = process.env.MAKEFUN_PROMPT_FILE ?? "";
  if (file === "") return "";
  try {
    return readFileSync(file, "utf8");
  } catch {
    return "";
  }
}

void main().catch((err) => {
  process.stderr.write(`entry: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
