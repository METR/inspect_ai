import { jest } from "@jest/globals";
import { resolveSample } from "../../state/sampleUtils";

// Minimal ChatMessage factories
const sysMsg = (id: string, content: string) => ({
  id,
  role: "system",
  content,
  source: "input",
  metadata: {},
});

const userMsg = (id: string, content: string) => ({
  id,
  role: "user",
  content,
  source: "input",
  metadata: {},
});

// Minimal ModelEvent factory
const modelEvent = (input: unknown[], input_refs?: string[] | null) => ({
  event: "model",
  model: "test-model",
  input,
  input_refs: input_refs ?? null,
  output: { choices: [] },
  timestamp: "2025-01-01T00:00:00Z",
  uuid: "test-uuid",
  span_id: null,
  working_start: null,
  metadata: {},
  pending: false,
  role: "assistant",
  tools: [],
  tool_choice: "auto",
  config: {},
  retries: null,
  error: null,
  traceback: null,
  traceback_ansi: null,
  cache: null,
  call: null,
  completed: null,
  working_time: null,
});

const toolEvent = (events: unknown[]) => ({
  event: "tool",
  type: "function",
  id: "tool-1",
  function: "test_fn",
  arguments: {},
  result: "ok",
  events,
  timestamp: "2025-01-01T00:00:00Z",
  uuid: "tool-uuid",
  span_id: null,
  working_start: null,
  metadata: {},
  pending: false,
  truncated: null,
  error: null,
  view: null,
  completed: null,
  working_time: null,
  agent: null,
  failed: null,
  message_id: null,
});

const subtaskEvent = (events: unknown[]) => ({
  event: "subtask",
  name: "test-subtask",
  type: "agent",
  input: {},
  result: {},
  events,
  timestamp: "2025-01-01T00:00:00Z",
  uuid: "subtask-uuid",
  span_id: null,
  working_start: null,
  metadata: {},
  pending: false,
  completed: null,
  working_time: null,
});

const makeSample = (overrides: Record<string, unknown> = {}) => ({
  id: "sample-1",
  epoch: 1,
  input: "test input",
  choices: null,
  target: "test target",
  sandbox: null,
  files: null,
  setup: null,
  messages: [],
  output: { choices: [] },
  scores: null,
  metadata: {},
  store: {},
  events: [],
  timelines: null,
  model_usage: {},
  started_at: null,
  completed_at: null,
  total_time: null,
  working_time: null,
  uuid: null,
  invalidation: null,
  error: null,
  error_retries: null,
  attachments: {},
  limit: null,
  ...overrides,
});

describe("resolveSample - message pool resolution", () => {
  test("resolves input_refs from message_pool into ModelEvent.input", () => {
    const sys = sysMsg("msg-1", "You are helpful.");
    const usr = userMsg("msg-2", "What is 2+2?");
    const pool = { "msg-1": sys, "msg-2": usr };

    const sample = makeSample({
      message_pool: pool,
      events: [modelEvent([], ["msg-1", "msg-2"])],
    });

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual([sys, usr]);
    expect((resolved.events[0] as any).input_refs).toBeUndefined();
  });

  test("resolves nested ModelEvents inside ToolEvent.events", () => {
    const sys = sysMsg("msg-1", "System prompt");
    const pool = { "msg-1": sys };

    const nested = modelEvent([], ["msg-1"]);
    const sample = makeSample({
      message_pool: pool,
      events: [toolEvent([nested])],
    });

    const resolved = resolveSample(sample);
    const innerEvents = (resolved.events[0] as any).events;
    expect(innerEvents[0].input).toEqual([sys]);
  });

  test("resolves nested ModelEvents inside SubtaskEvent.events", () => {
    const usr = userMsg("msg-1", "Hello");
    const pool = { "msg-1": usr };

    const nested = modelEvent([], ["msg-1"]);
    const sample = makeSample({
      message_pool: pool,
      events: [subtaskEvent([nested])],
    });

    const resolved = resolveSample(sample);
    const innerEvents = (resolved.events[0] as any).events;
    expect(innerEvents[0].input).toEqual([usr]);
  });

  test("skips missing refs with console.warn", () => {
    const sys = sysMsg("msg-1", "System");
    const pool = { "msg-1": sys };

    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const sample = makeSample({
      message_pool: pool,
      events: [modelEvent([], ["msg-1", "msg-missing"])],
    });

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual([sys]);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("msg-missing"),
    );
    warnSpy.mockRestore();
  });

  test("passes through sample unchanged when message_pool is empty", () => {
    const existingInput = [sysMsg("msg-1", "Hi")];
    const sample = makeSample({
      events: [modelEvent(existingInput)],
    });

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual(existingInput);
  });

  test("passes through sample unchanged when message_pool is absent", () => {
    const existingInput = [userMsg("msg-1", "Hi")];
    const sample = makeSample({
      events: [modelEvent(existingInput)],
    });
    delete (sample as any).message_pool;

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual(existingInput);
  });

  test("clears message_pool after resolution", () => {
    const pool = { "msg-1": sysMsg("msg-1", "Hello") };
    const sample = makeSample({
      message_pool: pool,
      events: [modelEvent([], ["msg-1"])],
    });

    const resolved = resolveSample(sample);
    expect((resolved as any).message_pool).toEqual({});
  });

  test("resolves pool before attachments so attachment refs in pool messages work", () => {
    const msgWithAttachment = userMsg("msg-1", "attachment://abc123");
    const pool = { "msg-1": msgWithAttachment };
    const attachments = { abc123: "resolved content" };

    const sample = makeSample({
      message_pool: pool,
      attachments,
      events: [modelEvent([], ["msg-1"])],
    });

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input[0].content).toBe(
      "resolved content",
    );
  });
});
