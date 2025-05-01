import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Progress } from "./ui/progress";
import { WS } from "../api/client";

export function SimulationModal({
  id,
  open,
  onOpenChange,
}: {
  id: string;
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const [logs, setLogs] = useState<string[]>([]);
  const [pct, setPct] = useState(0);
  const [gif, setGif] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;

    const ws = new WebSocket(`${WS}/ws/simulate/${id}`);

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      switch (msg.type) {
        case "progress":
          setPct(msg.value);
          break;
        case "log":
          setLogs((l) => [...l, msg.text]);
          break;
        case "finished":
          setGif(`${WS.replace(/^ws/, "http")}${msg.gifUrl}`);
          ws.close();
          break;
        case "error":
          setLogs((l) => [...l, `ERROR: ${msg.msg}`]);
          ws.close();
          break;
      }
    };

    ws.onerror = () => setLogs((l) => [...l, "Socket error"]);

    return () => ws.close();
  }, [open, id]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        {gif ? (
          <img src={gif} alt="Routed result" className="rounded-xl" />
        ) : (
          <>
            <Progress value={pct} className="mb-4" />
            <pre className="h-72 overflow-y-auto text-xs bg-muted p-2 rounded">
              {logs.join("\n")}
            </pre>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}