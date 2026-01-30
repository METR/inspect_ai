import clsx from "clsx";
import { FC } from "react";
import {
  ContentAudio,
  ContentImage,
  ContentMarkdown,
  ContentVideo,
  Format1,
  Format2,
  MediaEvent,
} from "../../../@types/log";
import { formatDateTime, toTitleCase } from "../../../utils/format";
import { ApplicationIcons } from "../../appearance/icons";
import { RenderedText } from "../../content/RenderedText";
import { EventPanel } from "./event/EventPanel";
import { formatTitle } from "./event/utils";
import styles from "./MediaEventView.module.css";
import { EventNode } from "./types";

interface MediaEventViewProps {
  eventNode: EventNode<MediaEvent>;
  className?: string | string[];
}

const MEDIA_ICONS: Record<string, string> = {
  image: "bi bi-image",
  audio: "bi bi-music-note-beamed",
  video: "bi bi-camera-video",
  markdown: "bi bi-markdown",
};

const MIME_TYPES: Record<string, string> = {
  mov: "video/quicktime",
  wav: "audio/wav",
  mp3: "audio/mpeg",
  mp4: "video/mp4",
  mpeg: "video/mpeg",
};

/**
 * Renders the MediaEventView component.
 */
export const MediaEventView: FC<MediaEventViewProps> = ({
  eventNode,
  className,
}) => {
  const event = eventNode.event;

  const renderMedia = () => {
    const content = event.content;
    switch (content.type) {
      case "image":
        return renderImage(content);
      case "audio":
        return renderAudio(content);
      case "video":
        return renderVideo(content);
      case "markdown":
        return renderMarkdown(content);
      default: {
        const exhaustiveCheck: never = content;
        throw new Error(`Unhandled content type: ${exhaustiveCheck}`);
      }
    }
  };

  const renderImage = (content: ContentImage) => {
    if (content.image.startsWith("data:")) {
      return (
        <img
          src={content.image}
          className={styles.image}
          alt={event.caption || "Media"}
        />
      );
    } else {
      return <code className={styles.url}>{content.image}</code>;
    }
  };

  const renderAudio = (content: ContentAudio) => {
    return (
      <audio controls className={styles.audio}>
        <source src={content.audio} type={mimeTypeForFormat(content.format)} />
      </audio>
    );
  };

  const renderVideo = (content: ContentVideo) => {
    return (
      <video controls className={styles.video}>
        <source src={content.video} type={mimeTypeForFormat(content.format)} />
      </video>
    );
  };

  const renderMarkdown = (content: ContentMarkdown) => {
    return (
      <RenderedText
        markdown={content.markdown}
        className={clsx(styles.markdown, "text-size-base")}
      />
    );
  };

  const getIcon = () => {
    return MEDIA_ICONS[event.content.type] ?? ApplicationIcons.info;
  };

  const getTitle = () => {
    const typeLabel = toTitleCase(event.content.type);
    if (event.source) {
      return `${typeLabel}: ${event.source}`;
    }
    return typeLabel;
  };

  return (
    <EventPanel
      eventNodeId={eventNode.id}
      depth={eventNode.depth}
      title={formatTitle(getTitle(), undefined, event.working_start)}
      className={className}
      subTitle={formatDateTime(new Date(event.timestamp))}
      icon={getIcon()}
    >
      <div className={styles.container}>
        {renderMedia()}
        {event.caption && (
          <div className={clsx(styles.caption, "text-style-secondary")}>
            {event.caption}
          </div>
        )}
      </div>
    </EventPanel>
  );
};

/**
 * Returns the MIME type for a given format.
 */
const mimeTypeForFormat = (format: Format1 | Format2): string => {
  return MIME_TYPES[format] ?? "video/mp4";
};
