import { useState, useEffect } from 'react';
import CircuitView from './CircuitView';
import LayoutView from './LayoutView';

interface CombinedCircuitViewProps {
  benchContent: string;
  defContent: string;
  coordinates?: [number, number][];
  gate_names?: string[];
}

export default function CombinedCircuitView({
  benchContent,
  defContent,
  coordinates,
  gate_names
}: CombinedCircuitViewProps) {
  const [view, setView] = useState<'circuit' | 'layout'>('circuit');
  const [highlightedInstance, setHighlightedInstance] = useState<string>();
  const [lefContent, setLefContent] = useState<string>();

  useEffect(() => {
    // Fetch LEF content when layout view is selected
    if (view === 'layout') {
      fetch('/api/layout/default/lef')
        .then(res => res.text())
        .then(content => setLefContent(content))
        .catch(err => console.error('Failed to fetch LEF:', err));
    }
  }, [view]);

  return (
    <div className="space-y-4">
      <div className="flex space-x-2">
        <button
          onClick={() => setView('circuit')}
          className={`px-4 py-2 rounded ${
            view === 'circuit'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          Circuit View
        </button>
        <button
          onClick={() => setView('layout')}
          className={`px-4 py-2 rounded ${
            view === 'layout'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700'
          }`}
        >
          Layout View
        </button>
      </div>

      {view === 'circuit' ? (
        <CircuitView
          content={benchContent}
          coordinates={coordinates}
          gate_names={gate_names}
        />
      ) : (
        <LayoutView
          defContent={defContent}
          lefContent={lefContent}
          highlightedInstance={highlightedInstance}
          onInstanceClick={(instance) => setHighlightedInstance(instance.name)}
        />
      )}
    </div>
  );
} 