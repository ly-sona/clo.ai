import { Card, CardContent } from "./ui/card";
import { ArrowDown, ArrowUp, FileBox, Eye } from "lucide-react";
import { Button } from "./ui/button";
import type { LayoutMeta } from "../pages/Home";
import LayoutModal from "./LayoutModal";

interface LayoutCardSimpleProps extends LayoutMeta {
  original_power?: number;
  original_delay?: number;
  new_delay?: number;
}

export default function LayoutCardSimple({ 
  id, 
  power, 
  thumb,
  optimized_circuit,
  coordinates,
  gate_names,
  original_power, 
  original_delay, 
  new_delay 
}: LayoutCardSimpleProps) {
  // Calculate percentage changes
  const powerChange = original_power ? ((power - original_power) / original_power) * 100 : 0;
  const delayChange = original_delay && new_delay ? ((new_delay - original_delay) / original_delay) * 100 : 0;

  // Prepare layout data for the modal
  const layoutData: LayoutMeta = {
    id,
    power,
    thumb,
    optimized_circuit,
    coordinates,
    gate_names
  };

  return (
    <LayoutModal 
      layout={layoutData}
      original_power={original_power}
      original_delay={original_delay}
      new_delay={new_delay}
      trigger={
        <Card className="hover:shadow-xl transition-all duration-300 cursor-pointer border border-slate-800/60 bg-gradient-to-b from-slate-900 to-slate-950 overflow-hidden group relative">
          {/* Hover gradient effect */}
          <div className="absolute inset-0 bg-gradient-to-tr from-indigo-800/5 via-transparent to-purple-800/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          
          <CardContent className="p-6">
            <div className="flex items-start gap-5">
              {/* File Icon - Enhanced */}
              <div className="bg-gradient-to-br from-indigo-900/70 to-indigo-700/50 rounded-lg p-3 shadow-md border border-indigo-700/30 group-hover:border-indigo-500/50 transition-colors">
                <FileBox size={32} className="text-indigo-200 group-hover:text-indigo-100 transition-colors" />
              </div>
              
              {/* Content */}
              <div className="flex-1 space-y-3">
                <h3 className="text-lg font-semibold truncate quantico-bold text-white group-hover:text-indigo-200 transition-colors">
                  {id}
                </h3>
                
                {/* Metrics - Enhanced grid layout */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  {/* Power metrics */}
                  <div className="bg-slate-900/40 rounded-md p-2 border border-slate-800/50">
                    <div className="text-white/70 text-xs mb-1">Power</div>
                    <div className="text-white font-medium">{power.toFixed(3)} mW</div>
                    <div className="flex items-center gap-1 mt-1">
                      {powerChange < 0 ? (
                        <ArrowDown className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <ArrowUp className="text-red-500 h-3.5 w-3.5" />
                      )}
                      <span className={powerChange < 0 ? "text-green-500 text-xs" : "text-red-500 text-xs"}>
                        {Math.abs(powerChange).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  {/* Delay metrics */}
                  <div className="bg-slate-900/40 rounded-md p-2 border border-slate-800/50">
                    <div className="text-white/70 text-xs mb-1">Delay</div>
                    <div className="text-white font-medium">{new_delay?.toFixed(2) ?? 'N/A'} ns</div>
                    <div className="flex items-center gap-1 mt-1">
                      {delayChange < 0 ? (
                        <ArrowDown className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <ArrowUp className="text-red-500 h-3.5 w-3.5" />
                      )}
                      <span className={delayChange < 0 ? "text-green-500 text-xs" : "text-red-500 text-xs"}>
                        {Math.abs(delayChange).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* View button - Enhanced */}
              <Button 
                variant="outline" 
                size="sm" 
                className="mt-1 bg-slate-800/60 border-slate-700/50 hover:bg-indigo-900/60 hover:border-indigo-700/50 text-white/80 hover:text-white transition-colors"
              >
                <Eye className="h-4 w-4 mr-1" />
                View
              </Button>
            </div>
          </CardContent>
        </Card>
      }
    />
  );
} 