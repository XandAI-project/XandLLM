import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, ChevronDown, ChevronRight, Brain } from "lucide-react";
import clsx from "clsx";
import type { Message as MessageType } from "../types";

// ── Think-block parser ────────────────────────────────────────────────────────

type Segment =
  | { type: "text"; content: string }
  | { type: "think"; content: string; closed: boolean };

function parseContent(raw: string): Segment[] {
  const segments: Segment[] = [];
  const OPEN = "<think>";
  const CLOSE = "</think>";
  let rest = raw;

  while (rest.length > 0) {
    const openIdx = rest.indexOf(OPEN);

    if (openIdx === -1) {
      // No more think tags — remainder is plain text
      if (rest.length > 0) segments.push({ type: "text", content: rest });
      break;
    }

    // Text before the opening tag
    if (openIdx > 0) {
      segments.push({ type: "text", content: rest.slice(0, openIdx) });
    }

    const afterOpen = rest.slice(openIdx + OPEN.length);
    const closeIdx = afterOpen.indexOf(CLOSE);

    if (closeIdx === -1) {
      // Still inside an open think block (streaming in progress)
      segments.push({ type: "think", content: afterOpen, closed: false });
      break;
    }

    // Fully closed think block
    segments.push({
      type: "think",
      content: afterOpen.slice(0, closeIdx),
      closed: true,
    });
    rest = afterOpen.slice(closeIdx + CLOSE.length);
  }

  return segments;
}

// ── Markdown renderer ─────────────────────────────────────────────────────────

function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const isBlock = className?.includes("language-");
          if (isBlock) {
            return (
              <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 overflow-x-auto my-2">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          }
          return (
            <code
              className="bg-gray-900 px-1.5 py-0.5 rounded text-blue-300 text-xs"
              {...props}
            >
              {children}
            </code>
          );
        },
      }}
    >
      {content || " "}
    </ReactMarkdown>
  );
}

// ── Think block component ─────────────────────────────────────────────────────

function ThinkBlock({
  content,
  closed,
  defaultOpen,
}: {
  content: string;
  closed: boolean;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="my-2 rounded-lg border border-amber-700/40 bg-amber-950/30 overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-amber-400 hover:bg-amber-900/20 transition-colors text-left"
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <Brain size={12} className="flex-shrink-0" />
        <span>{closed ? "Thinking" : "Thinking…"}</span>
        {!closed && (
          <span className="ml-auto flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce [animation-delay:0ms]" />
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce [animation-delay:150ms]" />
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce [animation-delay:300ms]" />
          </span>
        )}
      </button>

      {/* Body */}
      {open && (
        <div className="px-3 pb-3 pt-1 text-xs text-amber-200/70 leading-relaxed border-t border-amber-700/30">
          <div className="prose prose-sm max-w-none text-amber-200/80 [&_*]:text-amber-200/80">
            <MarkdownContent content={content.trim()} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Message component ────────────────────────────────────────────────────

interface Props {
  message: MessageType;
  showThink: boolean;
}

export default function Message({ message, showThink }: Props) {
  const isUser = message.role === "user";
  const segments = isUser ? null : parseContent(message.content);
  const hasThink = segments?.some((s) => s.type === "think") ?? false;

  return (
    <div
      className={clsx(
        "flex gap-3 px-4 py-4 group",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm mt-0.5",
          isUser ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-200"
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Bubble */}
      <div
        className={clsx(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-blue-600 text-white rounded-tr-sm"
            : "bg-gray-800 text-gray-100 rounded-tl-sm"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div>
            {/* Thinking segments */}
            {hasThink &&
              segments!.map((seg, i) => {
                if (seg.type === "think") {
                  return (
                    <ThinkBlock
                      key={i}
                      content={seg.content}
                      closed={seg.closed}
                      defaultOpen={showThink}
                    />
                  );
                }
                return null;
              })}

            {/* Text segments — rendered as markdown */}
            <div
              className={clsx(
                "prose prose-invert prose-sm max-w-none",
                message.streaming && !hasThink && "streaming-cursor"
              )}
            >
              {segments && segments.some((s) => s.type === "text") ? (
                segments.map((seg, i) =>
                  seg.type === "text" ? (
                    <MarkdownContent key={i} content={seg.content} />
                  ) : null
                )
              ) : (
                // No text segments yet (still thinking or empty)
                message.streaming ? null : (
                  <MarkdownContent content={message.content} />
                )
              )}
              {/* Streaming cursor after last text when still generating */}
              {message.streaming && <span className="streaming-cursor" />}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
