import { useState } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Progress } from "./ui/progress";
import { apiPostFile } from "../api/client";

export default function UploadDropzone({ onFinished }: { onFinished: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [pct, setPct] = useState(0);

  const handleUpload = async () => {
    if (!file) return;
    await apiPostFile("/optimize", file, setPct);
    setFile(null);
    setPct(0);
    onFinished();              // refresh gallery in Home
  };

  return (
    <Card className="w-full max-w-xl mx-auto p-8 flex flex-col items-center gap-4">
      <input
        id="schem"
        type="file"
        accept=".json,.yaml,.def,.lef"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="hidden"
      />
      <label htmlFor="schem" className="text-center cursor-pointer">
        {file ? file.name : "Drag / click to choose your schematic"}
      </label>

      {pct > 0 && <Progress value={pct} className="w-full" />}
      <Button disabled={!file || pct > 0} onClick={handleUpload} className="w-full">
        Optimize
      </Button>
    </Card>
  );
}