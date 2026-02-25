import { useState, useEffect, useCallback } from "react";
import { fetchHealth, fetchModels } from "../api";
import type { ServerStatus } from "../types";

const POLL_INTERVAL_MS = 10_000;

export function useServerInfo(baseUrl: string): ServerStatus & { refresh: () => void } {
  const [status, setStatus] = useState<ServerStatus>({ online: false, model: null });

  const check = useCallback(async () => {
    const online = await fetchHealth(baseUrl);
    if (!online) {
      setStatus({ online: false, model: null });
      return;
    }
    const models = await fetchModels(baseUrl);
    const model = models?.data?.[0]?.id ?? null;
    setStatus({ online: true, model });
  }, [baseUrl]);

  useEffect(() => {
    check();
    const id = setInterval(check, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [check]);

  return { ...status, refresh: check };
}
