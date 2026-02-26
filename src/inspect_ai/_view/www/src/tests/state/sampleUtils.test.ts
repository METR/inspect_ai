import { jest } from "@jest/globals";
import { resolveSample } from "../../state/sampleUtils";

// Minimal factories – only fields the resolution code dispatches on

const msg = (id: string, role: string, content: string) => ({
  id,
  role,
  content,
  source: "input",
  metadata: {},
});

const modelEvent = (input: unknown[], input_refs?: string[] | null) => ({
  event: "model",
  input,
  input_refs: input_refs ?? null,
});

const toolEvent = (events: unknown[]) => ({
  event: "tool",
  events,
});

const subtaskEvent = (events: unknown[]) => ({
  event: "subtask",
  events,
});

const makeSample = (overrides: Record<string, unknown> = {}) => ({
  input: "test input",
  messages: [],
  events: [],
  attachments: {},
  ...overrides,
});

describe("resolveSample - message pool resolution", () => {
  test("resolves input_refs from message_pool into ModelEvent.input", () => {
    const sys = msg("msg-1", "system", "You are helpful.");
    const usr = msg("msg-2", "user", "What is 2+2?");
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
    const sys = msg("msg-1", "system", "System prompt");
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
    const usr = msg("msg-1", "user", "Hello");
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
    const sys = msg("msg-1", "system", "System");
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
    const existingInput = [msg("msg-1", "system", "Hi")];
    const sample = makeSample({
      events: [modelEvent(existingInput)],
    });

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual(existingInput);
  });

  test("passes through sample unchanged when message_pool is absent", () => {
    const existingInput = [msg("msg-1", "user", "Hi")];
    const sample = makeSample({
      events: [modelEvent(existingInput)],
    });
    delete (sample as any).message_pool;

    const resolved = resolveSample(sample);
    expect((resolved.events[0] as any).input).toEqual(existingInput);
  });

  test("clears message_pool after resolution", () => {
    const pool = { "msg-1": msg("msg-1", "system", "Hello") };
    const sample = makeSample({
      message_pool: pool,
      events: [modelEvent([], ["msg-1"])],
    });

    const resolved = resolveSample(sample);
    expect((resolved as any).message_pool).toEqual({});
  });

  test("resolves pool before attachments so attachment refs in pool messages work", () => {
    const msgWithAttachment = msg("msg-1", "user", "attachment://abc123");
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

  test("resolves ModelEvents nested multiple levels deep", () => {
    const m = msg("msg-1", "user", "Deep");
    const pool = { "msg-1": m };

    const sample = makeSample({
      message_pool: pool,
      events: [subtaskEvent([toolEvent([modelEvent([], ["msg-1"])])])],
    });

    const resolved = resolveSample(sample);
    const innerModel = (resolved.events[0] as any).events[0].events[0];
    expect(innerModel.input).toEqual([m]);
  });
});
