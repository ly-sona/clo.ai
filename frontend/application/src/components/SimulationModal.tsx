// src/components/SimulationModal.tsx
import { useEffect, useState } from "react";
import { Dialog, DialogContent } from "./ui/dialog";
import { Progress } from "./ui/progress";

export function SimulationModal({id, open, onOpenChange}:{id:string;open:boolean;onOpenChange:(o:boolean)=>void;}) {
  const [logs, setLogs] = useState<string[]>([]);
  const [pct, setPct]   = useState(0);
  const [gif, setGif]   = useState<string|null>(null);

  useEffect(()=>{
    if (!open) return;
    const ws = new WebSocket(`${import.meta.env.VITE_WS_ROOT}/ws/simulate/${id}`);
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "progress") setPct(msg.value);
      else if (msg.type === "log") setLogs(l => [...l, msg.text]);
      else if (msg.type === "finished") setGif(msg.gifUrl);
    };
    return ()=>ws.close();
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