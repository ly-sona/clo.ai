import { useState } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Progress } from "./ui/progress";
import { apiPostFile, apiGetJson } from "../api/client";
import type { LayoutMeta } from "../pages/Home";
import { Upload, Sparkles } from "lucide-react";
import CircuitView from "./CircuitView";

export default function UploadDropzone({ onFinished }: { onFinished: (layout?: LayoutMeta) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      // Read and display the file content
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target?.result as string);
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setError(null);
    try {
      const job_id = await apiPostFile("/optimize", file, setPct);
      setFile(null);
      setFileContent("");
      setPct(0);
      // Poll for the new layout
      let found = false;
      let tries = 0;
      let newLayout: LayoutMeta | undefined = undefined;
      while (!found && tries < 30) { // try for up to 30 seconds
        const layouts = await apiGetJson<LayoutMeta[]>("/layouts");
        newLayout = layouts.find(l => l.id === job_id);
        if (newLayout) {
          found = true;
          break;
        } else {
          await new Promise(res => setTimeout(res, 1000));
          tries++;
        }
      }
      if (found && newLayout) {
        onFinished(newLayout);
      } else {
        setError("Optimization timed out.");
        onFinished();
      }
    } catch (err) {
      setError("Failed to upload file. Please try again.");
      console.error("Upload error:", err);
    }
  };

  return (
    <Card className="w-full md:w-2/5 p-8 flex flex-col items-center gap-4 mx-auto">
      <input
        id="schem"
        type="file"
        accept=".bench,.json,.yaml,.def,.lef"
        onChange={handleFileChange}
        className="hidden"
      />
      <label htmlFor="schem" className="text-center cursor-pointer w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-slate-800 text-indigo-300 font-medium border border-indigo-400 hover:bg-indigo-900 hover:text-indigo-200 transition-colors mb-2 shadow focus:outline-none focus:ring-2 focus:ring-indigo-400">
        <Upload className="w-5 h-5" />
        <span className="truncate">{file ? file.name : "Choose your schematic"}</span>
      </label>

      {fileContent && (
        <div className="w-full mt-4">
          <h3 className="text-lg font-semibold mb-2 text-white">Circuit Preview</h3>
          <div className="w-full h-[500px] border rounded-lg overflow-hidden">
            <CircuitView content={fileContent} />
          </div>
        </div>
      )}

      {error && (
        <div className="text-red-500 text-sm">{error}</div>
      )}

      {pct > 0 && <Progress value={pct} className="w-full" />}
      <Button 
        disabled={!file || pct > 0} 
        onClick={handleUpload} 
        className="animated-gradient w-full flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-300 via-sky-200 to-purple-300 text-slate-900 font-semibold py-3 rounded-lg shadow focus:outline-none focus:ring-2 focus:ring-indigo-300/50"
      >
        <Sparkles className="w-5 h-5" />
        {pct > 0 ? "Uploading..." : "Optimize"}
      </Button>
    </Card>
  );
}