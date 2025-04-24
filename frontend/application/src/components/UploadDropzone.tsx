// src/components/UploadDropzone.tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Progress } from "./ui/progress";
import { apiPostFile } from "../api/client";

export default function UploadDropzone() {
  const [file, setFile] = useState<File | null>(null);
  const [pct, setPct]   = useState(0);
  const nav = useNavigate();

  const handleUpload = async () => {
    if (!file) return;
    const id = await apiPostFile("/optimize", file, setPct); // updates pct
    nav(`/?job=${id}`);     // on success go back to gallery, query param selects new job
  };

  return (
    <Card className="w-full max-w-xl mx-auto p-8 flex flex-col items-center gap-4">
      <input
        type="file"
        accept=".json,.yaml,.def,.lef"
        onChange={e => setFile(e.target.files?.[0] ?? null)}
        className="hidden" id="schem"
      />
      <label htmlFor="schem" className="text-center cursor-pointer">
        {file ? file.name : "Drag / click to choose your schematic"}
      </label>

      {pct > 0 && <Progress value={pct} className="w-full" />}
      <Button disabled={!file} onClick={handleUpload} className="w-full">
        Optimize
      </Button>
    </Card>
  );
}