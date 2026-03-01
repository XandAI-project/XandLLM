import type { ServerStatus } from "../types";

interface Props {
  status: ServerStatus;
}

export default function ModelBadge({ status }: Props) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 max-w-[240px]">
      <span
        className={`w-2 h-2 rounded-full flex-shrink-0 ${
          status.online ? "bg-green-400" : "bg-red-500"
        }`}
      />
      <span className="text-xs text-gray-300 truncate">
        {status.online
          ? status.model ?? "Connected"
          : "Server offline"}
      </span>
    </div>
  );
}
