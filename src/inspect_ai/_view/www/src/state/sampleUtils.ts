import { EvalSample } from "../@types/log";
import { resolveAttachments } from "../utils/attachments";

type ChatMessage = EvalSample["messages"][number];
type MessagePool = Record<string, ChatMessage>;

/**
 * Resolve input_refs in a flat event list against the message pool.
 * Recurses into ToolEvent.events and SubtaskEvent.events.
 */
const resolveEventInputRefs = (events: any[], pool: MessagePool): any[] => {
  return events.map((event) => {
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
      const { input_refs, ...rest } = event;
      return { ...rest, input: resolvedInput };
    }

    if (
      (event.event === "tool" || event.event === "subtask") &&
      Array.isArray(event.events)
    ) {
      return { ...event, events: resolveEventInputRefs(event.events, pool) };
    }

    return event;
  });
};

/**
 * Resolve message pool references in a sample's events.
 * Returns the sample unchanged if no message_pool is present.
 */
const resolveMessagePool = (sample: any): any => {
  const pool: MessagePool | undefined = sample.message_pool;
  if (!pool || Object.keys(pool).length === 0) {
    return sample;
  }

  return {
    ...sample,
    events: resolveEventInputRefs(sample.events || [], pool),
    message_pool: {},
  };
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

  sample.attachments = sample.attachments || {};
  sample.input = resolveAttachments(sample.input, sample.attachments);
  sample.messages = resolveAttachments(sample.messages, sample.attachments);
  sample.events = resolveAttachments(sample.events, sample.attachments);
  sample.attachments = {};
  return sample;
};
