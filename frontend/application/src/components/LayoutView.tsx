import { useEffect, useState } from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { apiGetDef, apiGetLef } from "../api/client";
import DefView from "./DefView";

interface Instance {
  name: string;
  cell: string;
  x: number;
  y: number;
  orient: string;
}

interface Net {
  name: string;
  pins: { inst: string; pin: string }[];
}

interface CellInfo {
  name: string;
  size: {
    width: number;
    height: number;
  };
  pins: {
    name: string;
    x: number;
    y: number;
  }[];
}

interface LayoutViewProps {
  layoutId: string;
}

function parseDef(content: string): { components: Instance[]; nets: Net[] } {
  const components: Instance[] = [];
  const nets: Net[] = [];
  let currentNet: Net | null = null;

  const lines = content.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Parse COMPONENTS section
    if (trimmed.startsWith('- ')) {
      const match = trimmed.match(/- (\S+) (\S+) \+ PLACED \( (\d+) (\d+) \) (\S+)/);
      if (match) {
        const [, name, cell, x, y, orient] = match;
        components.push({
          name,
          cell,
          x: parseInt(x),
          y: parseInt(y),
          orient
        });
      }
    }

    // Parse NETS section
    if (trimmed.startsWith('NETS')) {
      currentNet = null;
    } else if (trimmed.startsWith('- ')) {
      const netName = trimmed.match(/- (\S+)/)?.[1];
      if (netName) {
        currentNet = { name: netName, pins: [] };
        nets.push(currentNet);
      }
    } else if (currentNet && trimmed.startsWith('(')) {
      const match = trimmed.match(/\( (\S+) (\S+) \)/);
      if (match) {
        const [, inst, pin] = match;
        currentNet.pins.push({ inst, pin });
      }
    }
  }

  return { components, nets };
}

function parseLef(content: string): Record<string, CellInfo> {
  const cells: Record<string, CellInfo> = {};
  let currentCell: string | null = null;
  
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    
    if (trimmed.startsWith('MACRO')) {
      currentCell = trimmed.split()[1];
      cells[currentCell] = {
        name: currentCell,
        size: { width: 0, height: 0 },
        pins: []
      };
    } else if (currentCell && trimmed.includes('SIZE')) {
      const parts = trimmed.split();
      const widthIdx = parts.indexOf('SIZE') + 1;
      cells[currentCell].size.width = parseFloat(parts[widthIdx]);
      cells[currentCell].size.height = parseFloat(parts[widthIdx + 2]);
    } else if (currentCell && trimmed.startsWith('PIN')) {
      const pinName = trimmed.split()[1];
      cells[currentCell].pins.push({
        name: pinName,
        x: 0,
        y: 0
      });
    }
  }
  
  return cells;
}

export default function LayoutView({ layoutId }: LayoutViewProps) {
  const [defContent, setDefContent] = useState<string>("");
  const [lefContent, setLefContent] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
      try {
        const [def, lef] = await Promise.all([
          apiGetDef(layoutId),
          apiGetLef()
        ]);
        setDefContent(def);
        setLefContent(lef);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load layout");
      }
    };
    fetchContent();
  }, [layoutId]);

  if (error) {
    return <div className="text-red-500">{error}</div>;
  }

  if (!defContent || !lefContent) {
    return <div>Loading...</div>;
  }

  return <DefView defContent={defContent} lefContent={lefContent} />;
} 