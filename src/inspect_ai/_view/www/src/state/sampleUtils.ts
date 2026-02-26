import { EvalSample, Events } from "../@types/log";
import { resolveAttachments } from "../utils/attachments";

type SampleEvent = Events[number];
type ChatMessage = EvalSample["messages"][number];
type MessagePool = Record<string, ChatMessage>;

/**
 * Resolve input_refs in a flat event list against the message pool.
 * Recurses into ToolEvent.events and SubtaskEvent.events.
 */
const resolveEventInputRefs = (
  events: SampleEvent[],
  pool: MessagePool,
): SampleEvent[] => {
  return events.map((event): SampleEvent => {
    if (event.event === "model" && Array.isArray(event.input_refs)) {
      const resolvedInput: ChatMessage[] = [];
      for (const ref of event.input_refs) {
        const msg = pool[ref];
        if (msg !== undefined) {
          resolvedInput.push(msg);
        } else {
          console.warn(`Message pool ref '${ref}' not found`);
        }
      }
      // Destructure to drop input_refs, then replace input with resolved messages
      const { input_refs, ...rest } = event;
      return { ...rest, input: resolvedInput };
    }

    if (
      (event.event === "tool" || event.event === "subtask") &&
      Array.isArray(event.events)
    ) {
      // ToolEvent.events and SubtaskEvent.events are typed as unknown[] in the
      // generated schema, but at runtime contain the same event union. The casts
      // bridge the generated schema's loose nested-event types.
      return {
        ...event,
        events: resolveEventInputRefs(
          event.events as SampleEvent[],
          pool,
        ) as unknown[],
      };
    }

    return event;
  });
};

/**
 * Resolve message pool references in a sample's events.
 * Returns the sample unchanged if no message_pool is present.
 */
const resolveMessagePool = (sample: EvalSample): EvalSample => {
  const pool: MessagePool | undefined = sample.message_pool;
  if (!pool || Object.keys(pool).length === 0) {
    return sample;
  }

  return {
    ...sample,
    events: resolveEventInputRefs(sample.events, pool),
    message_pool: {},
  };
};

type CallMessagePool = Record<string, unknown>;

/**
 * Resolve request_message_refs in model events against the call_message_pool.
 * Restores call.request[key] from the pool and clears the refs.
 */
const resolveEventCallRefs = (
  events: SampleEvent[],
  pool: CallMessagePool,
): SampleEvent[] => {
  return events.map((event): SampleEvent => {
    if (event.event === "model" && event.call) {
      const call = event.call as unknown as Record<string, unknown>;
      const refs = call.request_message_refs as string[] | undefined;
      if (Array.isArray(refs)) {
        const msgKey = (call.request_message_key as string) || "messages";
        const msgs: unknown[] = [];
        for (const ref of refs) {
          if (ref in pool) {
            msgs.push(pool[ref]);
          }
        }
        const request = { ...(call.request as Record<string, unknown>) };
        request[msgKey] = msgs;
        return {
          ...event,
          call: {
            ...event.call,
            request,
            request_message_refs: undefined,
            request_message_key: undefined,
          },
        } as unknown as SampleEvent;
      }
    }

    if (
      (event.event === "tool" || event.event === "subtask") &&
      Array.isArray(event.events)
    ) {
      return {
        ...event,
        events: resolveEventCallRefs(
          event.events as SampleEvent[],
          pool,
        ) as unknown[],
      };
    }

    return event;
  });
};

const resolveCallMessagePool = (sample: EvalSample): EvalSample => {
  const pool = (sample as unknown as Record<string, unknown>)
    .call_message_pool as CallMessagePool | undefined;
  if (!pool || Object.keys(pool).length === 0) {
    return sample;
  }

  return {
    ...sample,
    events: resolveEventCallRefs(sample.events, pool),
    call_message_pool: {},
  } as EvalSample;
};

/**
 * Migrates and resolves attachments for a sample
 */
export const resolveSample = (sample: any): EvalSample => {
  sample = { ...sample };

  // Migrates old versions of samples to the new structure
  if (sample.transcript) {
    sample.events = sample.transcript.events;
    sample.attachments = sample.transcript.content;
  }

  // Resolve message pool refs BEFORE attachments (pool messages may
  // contain attachment:// refs that need resolving in the next step)
  sample = resolveMessagePool(sample);
  sample = resolveCallMessagePool(sample);

  sample.attachments = sample.attachments || {};
  sample.input = resolveAttachments(sample.input, sample.attachments);
  sample.messages = resolveAttachments(sample.messages, sample.attachments);
  sample.events = resolveAttachments(sample.events, sample.attachments);
  sample.attachments = {};
  return sample;
};
