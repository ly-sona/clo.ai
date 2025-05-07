import React from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import type { OptimizedCircuit as OptimizedCircuitType } from '../api/circuit';

interface OptimizedCircuitProps {
  circuit: OptimizedCircuitType;
  onDownload: (layoutId: string) => Promise<void>;
}

export function OptimizedCircuit({ circuit, onDownload }: OptimizedCircuitProps) {
  return (
    <Card className="p-6 space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold">Optimized Circuit</h2>
        <p className="text-gray-500">Job ID: {circuit.job_id}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {circuit.layouts.map((layout) => (
          <Card key={layout.id} className="p-4 space-y-4">
            <div className="aspect-video relative bg-gray-100 rounded-lg overflow-hidden">
              {layout.thumb && (
                <img
                  src={layout.thumb}
                  alt={`Layout ${layout.id}`}
                  className="w-full h-full object-contain"
                />
              )}
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium">Power: {layout.power.toFixed(2)}</span>
                {layout.wns && <span className="text-sm">WNS: {layout.wns}</span>}
              </div>
              
              {layout.cells && (
                <div className="text-sm text-gray-500">
                  Cells: {layout.cells}
                </div>
              )}

              <Button
                onClick={() => onDownload(layout.id)}
                className="w-full"
              >
                Download Layout
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </Card>
  );
} 