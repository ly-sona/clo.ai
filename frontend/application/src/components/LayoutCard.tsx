import { Card, CardContent, CardHeader } from "./ui/card";
import { useState } from "react";
import { ArrowDown, ArrowUp, DownloadIcon, EyeIcon } from "lucide-react";
import { Button } from "./ui/button";
import CircuitView from "./CircuitView";

interface Props {
  id: string;
  thumb?: string;   // now matches backend, made optional
  power: number;
  optimized_circuit?: string;
  original_power?: number;
  original_delay?: number;
  new_delay?: number;
  coordinates?: [number, number][];
  gate_names?: string[];
  isInModal?: boolean; // New prop to adjust styling when inside modal
}

export default function LayoutCard({ 
  id, 
  power, 
  optimized_circuit, 
  original_power, 
  original_delay, 
  new_delay, 
  coordinates, 
  gate_names,
  isInModal = false // Default to false
}: Props) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [showBench, setShowBench] = useState(false);
  const [benchText, setBenchText] = useState<string | null>(null);
  const [loadingBench, setLoadingBench] = useState(false);
  const [optimizedBenchText, setOptimizedBenchText] = useState<string | null>(null);
  const [loadingOptimizedBench, setLoadingOptimizedBench] = useState(false);

  // Calculate percentage changes
  const powerChange = original_power ? ((power - original_power) / original_power) * 100 : 0;
  const delayChange = original_delay && new_delay ? ((new_delay - original_delay) / original_delay) * 100 : 0;

  const handleShowBench = async () => {
    setShowBench(!showBench);
    if (!benchText && !showBench) {
      setLoadingBench(true);
      try {
        const res = await fetch(`/layout/${id}/bench`);
        if (!res.ok) throw new Error("Failed to fetch .bench file");
        const text = await res.text();
        setBenchText(text);
      } catch (e) {
        setBenchText("Error loading .bench file");
      } finally {
        setLoadingBench(false);
      }
    }
  };

  const handleDownloadOptimizedBench = async () => {
    if (!optimizedBenchText) {
      setLoadingOptimizedBench(true);
      try {
        const res = await fetch(`/layout/${id}/optimized-bench`);
        if (!res.ok) throw new Error("Failed to fetch optimized .bench file");
        const text = await res.text();
        setOptimizedBenchText(text);
        
        // Create and trigger download
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${id}_optimized.bench`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Error downloading optimized bench file:", e);
      } finally {
        setLoadingOptimizedBench(false);
      }
    } else {
      // If we already have the text, just download it
      const blob = new Blob([optimizedBenchText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${id}_optimized.bench`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Adjust heights based on whether the component is in a modal
  const circuitHeight = isInModal ? "65vh" : "500px";
  
  // Determine column layout based on context
  const columnLayout = isInModal ? "flex-row" : "flex-col lg:flex-row";
  const mainColumnWidth = isInModal ? "w-[75%]" : "w-full lg:w-3/4";
  const sideColumnWidth = isInModal ? "w-[25%]" : "w-full lg:w-1/4";

  return (
    <Card className={`transition-shadow duration-200 w-full mx-auto ${isInModal ? 'mb-0 shadow-none border-0' : 'mb-8 border border-slate-200 bg-white shadow-md'}`}>
      {!isInModal && (
        <CardHeader className="bg-gradient-to-r from-indigo-50 to-slate-50 border-b border-slate-200">
          <h3 className="text-lg font-semibold truncate quantico-bold text-slate-800">{id}</h3>
        </CardHeader>
      )}
      <CardContent className={isInModal ? "p-2" : "p-6"}>
        <div className={`flex ${columnLayout} gap-5`}>
          {/* Circuit View - Left Side */}
          <div className={mainColumnWidth}>
            {optimized_circuit ? (
              <div className={`w-full h-[${circuitHeight}] border border-slate-200 rounded-lg overflow-hidden shadow-inner bg-white`}>
                <CircuitView 
                  content={optimized_circuit} 
                  coordinates={coordinates} 
                  gate_names={gate_names} 
                  maxNodes={isInModal ? 1500 : 800} 
                />
              </div>
            ) : (
              <div className={`w-full h-[${circuitHeight}] border border-slate-200 rounded-lg flex items-center justify-center bg-slate-50`}>
                <p className="text-slate-500">No circuit data available</p>
              </div>
            )}
          </div>

          {/* Metrics Display - Right Side */}
          <div className={`${sideColumnWidth} flex flex-col`}>
            <div className="space-y-4">
              {/* Optimized metrics */}
              <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
                <div className="text-base font-semibold text-slate-800 mb-2">Optimized</div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Power:</span>
                    <span className="font-medium text-slate-900">{power.toFixed(3)} mW</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Delay:</span>
                    <span className="font-medium text-slate-900">{new_delay?.toFixed(2) ?? 'N/A'} ns</span>
                  </div>
                </div>
              </div>

              {/* Percent metrics */}
              <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
                <div className="text-base font-semibold text-slate-800 mb-2">Percent Metrics</div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Power:</span>
                    <div className="flex items-center gap-1">
                      {powerChange < 0 ? (
                        <ArrowDown className="text-green-600 h-4 w-4" />
                      ) : (
                        <ArrowUp className="text-red-600 h-4 w-4" />
                      )}
                      <span className={powerChange < 0 ? "text-green-600 font-medium" : "text-red-600 font-medium"}>
                        {Math.abs(powerChange).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Delay:</span>
                    <div className="flex items-center gap-1">
                      {delayChange < 0 ? (
                        <ArrowDown className="text-green-600 h-4 w-4" />
                      ) : (
                        <ArrowUp className="text-red-600 h-4 w-4" />
                      )}
                      <span className={delayChange < 0 ? "text-green-600 font-medium" : "text-red-600 font-medium"}>
                        {Math.abs(delayChange).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Original metrics (conditional) */}
              {showOriginal && (
                <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
                  <div className="text-base font-semibold text-slate-800 mb-2">Original</div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-600">Power:</span>
                      <span className="font-medium text-slate-900">{original_power?.toFixed(3) ?? 'N/A'} mW</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-600">Delay:</span>
                      <span className="font-medium text-slate-900">{original_delay?.toFixed(2) ?? 'N/A'} ns</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Action buttons */}
            <div className="flex flex-col gap-2 mt-auto w-full pt-4">
              <Button
                variant="outline"
                onClick={() => setShowOriginal(!showOriginal)}
                size={isInModal ? "sm" : "default"}
                className="w-full border-indigo-200 hover:bg-indigo-50 text-indigo-700 hover:text-indigo-800"
              >
                {showOriginal ? "Hide Original" : "Show Original"}
              </Button>

              <Button
                variant="outline"
                onClick={handleShowBench}
                size={isInModal ? "sm" : "default"}
                className="w-full border-indigo-200 hover:bg-indigo-50 text-indigo-700 hover:text-indigo-800"
              >
                {showBench ? "Hide Original Circuit" : "Show Original Circuit"}
              </Button>
              
              <Button
                variant="outline"
                onClick={handleDownloadOptimizedBench}
                size={isInModal ? "sm" : "default"}
                className="w-full border-indigo-200 hover:bg-indigo-50 text-indigo-700 hover:text-indigo-800"
                disabled={loadingOptimizedBench}
              >
                {loadingOptimizedBench ? "Loading..." : (
                  <div className="flex items-center gap-2">
                    <DownloadIcon className="h-4 w-4" />
                    <span>Download Optimized Circuit</span>
                  </div>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Original Circuit View (when expanded) */}
        {showBench && (
          <div className="w-full mt-6">
            {loadingBench ? (
              <div className="text-center text-slate-500">Loading...</div>
            ) : benchText ? (
              <div className={`w-full h-[${circuitHeight}] border border-slate-200 rounded-lg overflow-hidden shadow-inner bg-white`}>
                <CircuitView content={benchText} maxNodes={isInModal ? 1500 : 800} />
              </div>
            ) : (
              <div className="text-center text-slate-500">No circuit data available</div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}