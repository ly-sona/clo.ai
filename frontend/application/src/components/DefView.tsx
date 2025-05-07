import { useEffect, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

interface DefViewProps {
  defContent: string;
  lefContent: string;
}

interface Cell {
  name: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Pin {
  name: string;
  x: number;
  y: number;
  direction: string;
}

interface Net {
  name: string;
  pins: string[];
}

function parseDef(content: string): { cells: Cell[], nets: Net[] } {
  const cells: Cell[] = [];
  const nets: Net[] = [];
  let currentNet: Net | null = null;
  let inComponents = false;
  let inNets = false;

  const lines = content.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Track sections
    if (trimmed === 'COMPONENTS') {
      inComponents = true;
      inNets = false;
      continue;
    } else if (trimmed === 'NETS') {
      inComponents = false;
      inNets = true;
      continue;
    } else if (trimmed === 'END COMPONENTS') {
      inComponents = false;
      continue;
    } else if (trimmed === 'END NETS') {
      inNets = false;
      continue;
    }

    // Parse cell instances
    if (inComponents && trimmed.startsWith('- ')) {
      const cellMatch = trimmed.match(/- (\w+) \+ PLACED \( (\d+) (\d+) \) (\w+) ;$/);
      if (cellMatch) {
        const [_, name, x, y, type] = cellMatch;
        cells.push({
          name,
          type,
          x: parseInt(x),
          y: parseInt(y),
          width: 0, // Will be filled from LEF
          height: 0 // Will be filled from LEF
        });
      }
      continue;
    }

    // Parse nets
    if (inNets) {
      if (trimmed.startsWith('- ')) {
        if (currentNet) {
          nets.push(currentNet);
        }
        const netName = trimmed.match(/- (\w+)/)?.[1];
        if (netName) {
          currentNet = {
            name: netName,
            pins: []
          };
        }
      } else if (currentNet && trimmed.startsWith('(')) {
        const pinMatch = trimmed.match(/\( (\w+) \)/);
        if (pinMatch) {
          currentNet.pins.push(pinMatch[1]);
        }
      }
    }
  }

  if (currentNet) {
    nets.push(currentNet);
  }

  return { cells, nets };
}

function parseLef(content: string): Record<string, { width: number, height: number, pins: Pin[] }> {
  const macros: Record<string, { width: number, height: number, pins: Pin[] }> = {};
  let currentMacro: string | null = null;
  let currentPin: string | null = null;
  let currentDirection: string = 'UNKNOWN';

  const lines = content.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Parse macro definition
    const macroMatch = trimmed.match(/^MACRO (\w+)$/);
    if (macroMatch) {
      currentMacro = macroMatch[1];
      macros[currentMacro] = { width: 0, height: 0, pins: [] };
      continue;
    }

    if (currentMacro) {
      // Parse size
      const sizeMatch = trimmed.match(/^\s+SIZE ([\d.]+) BY ([\d.]+) ;$/);
      if (sizeMatch) {
        macros[currentMacro].width = parseFloat(sizeMatch[1]);
        macros[currentMacro].height = parseFloat(sizeMatch[2]);
        continue;
      }

      // Parse pin
      const pinMatch = trimmed.match(/^\s+PIN (\w+)$/);
      if (pinMatch) {
        currentPin = pinMatch[1];
        currentDirection = 'UNKNOWN';
        continue;
      }

      // Parse pin direction
      if (currentPin && trimmed.includes('DIRECTION')) {
        const dirMatch = trimmed.match(/DIRECTION\s+(\w+)\s*;/);
        if (dirMatch) {
          currentDirection = dirMatch[1];
        }
      }

      if (currentPin) {
        const rectMatch = trimmed.match(/^\s+RECT ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ;$/);
        if (rectMatch) {
          const [_, x1, y1, x2, y2] = rectMatch;
          macros[currentMacro].pins.push({
            name: currentPin,
            x: (parseFloat(x1) + parseFloat(x2)) / 2,
            y: (parseFloat(y1) + parseFloat(y2)) / 2,
            direction: currentDirection
          });
          currentPin = null;
        }
      }
    }
  }

  return macros;
}

export default function DefView({ defContent, lefContent }: DefViewProps) {
  const [cells, setCells] = useState<Cell[]>([]);
  const [nets, setNets] = useState<Net[]>([]);
  const [macros, setMacros] = useState<Record<string, { width: number, height: number, pins: Pin[] }>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      const { cells: parsedCells, nets: parsedNets } = parseDef(defContent);
      const parsedMacros = parseLef(lefContent);

      // Update cell dimensions from LEF
      const updatedCells = parsedCells.map(cell => ({
        ...cell,
        width: parsedMacros[cell.type]?.width || 1.4,
        height: parsedMacros[cell.type]?.height || 1.0
      }));

      setCells(updatedCells);
      setNets(parsedNets);
      setMacros(parsedMacros);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse layout files");
    }
  }, [defContent, lefContent]);

  if (error) {
    return <div className="text-red-500 p-4">{error}</div>;
  }

  if (cells.length === 0) {
    return <div className="p-4">No cells found in layout</div>;
  }

  // Calculate bounds for scaling
  const bounds = cells.reduce((acc, cell) => ({
    minX: Math.min(acc.minX, cell.x),
    minY: Math.min(acc.minY, cell.y),
    maxX: Math.max(acc.maxX, cell.x + cell.width),
    maxY: Math.max(acc.maxY, cell.y + cell.height)
  }), { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity });

  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  const scale = Math.min(800 / width, 600 / height);

  return (
    <div className="h-[600px] w-full border rounded-lg bg-gray-50">
      <TransformWrapper
        initialScale={1}
        minScale={0.1}
        maxScale={4}
        centerOnInit
      >
        <TransformComponent>
          <svg
            width={width * scale}
            height={height * scale}
            viewBox={`${bounds.minX} ${bounds.minY} ${width} ${height}`}
            className="bg-white"
          >
            {/* Draw cells */}
            {cells.map(cell => (
              <g key={cell.name}>
                <rect
                  x={cell.x}
                  y={cell.y}
                  width={cell.width}
                  height={cell.height}
                  fill="#60a5fa"
                  stroke="#1e40af"
                  strokeWidth={0.1}
                />
                <text
                  x={cell.x + cell.width/2}
                  y={cell.y + cell.height/2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="white"
                  fontSize={0.5}
                >
                  {cell.name}
                </text>
              </g>
            ))}

            {/* Draw nets */}
            {nets.map(net => (
              <g key={net.name}>
                {net.pins.map((pin, i) => {
                  const cell = cells.find(c => c.name === pin);
                  if (!cell) return null;
                  const macro = macros[cell.type];
                  const pinInfo = macro?.pins[0]; // Use first pin for now
                  return (
                    <circle
                      key={`${net.name}-${pin}`}
                      cx={cell.x + (pinInfo?.x || cell.width/2)}
                      cy={cell.y + (pinInfo?.y || cell.height/2)}
                      r={0.2}
                      fill="#f87171"
                    />
                  );
                })}
              </g>
            ))}
          </svg>
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
} 