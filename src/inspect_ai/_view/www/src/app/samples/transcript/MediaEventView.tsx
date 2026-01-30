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
import { formatDateTime } from "../../../utils/format";
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
      default:
        return null;
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
    switch (event.content.type) {
      case "image":
        return "bi bi-image";
      case "audio":
        return "bi bi-music-note-beamed";
      case "video":
        return "bi bi-camera-video";
      case "markdown":
        return "bi bi-markdown";
      default:
        return ApplicationIcons.info;
    }
  };

  const getTitle = () => {
    const typeLabel =
      event.content.type.charAt(0).toUpperCase() + event.content.type.slice(1);
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
  switch (format) {
    case "mov":
      return "video/quicktime";
    case "wav":
      return "audio/wav";
    case "mp3":
      return "audio/mpeg";
    case "mp4":
      return "video/mp4";
    case "mpeg":
      return "video/mpeg";
    default:
      return "video/mp4";
  }
};
