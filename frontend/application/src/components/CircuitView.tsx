import { useEffect, useState, useCallback } from "react";
import ReactFlow, { 
  Background, 
  Controls, 
  Node, 
  Edge, 
  NodeProps, 
  Handle,
  Position,
  Panel,
  useReactFlow,
  ReactFlowProvider
} from "reactflow";
import dagre from "@dagrejs/dagre";
import "reactflow/dist/style.css";

interface CircuitNode {
  id: string;
  type: string;
  inputs: string[];
  size?: number;
  group?: string; // For logical grouping
}

function parseBench(content: string): CircuitNode[] {
  const nodes: CircuitNode[] = [];
  const defined = new Set<string>();
  const referenced = new Set<string>();
  
  for (const line of content.split(/\r?\n/)) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;

    // Skip pure comment lines
    if (trimmedLine.startsWith('#')) continue;

    // Extract size from comments like "# size=0.97"
    let size: number | undefined;
    const sizeMatch = trimmedLine.match(/#.*size=(\d+\.\d+)/);
    if (sizeMatch) {
      size = parseFloat(sizeMatch[1]);
    }

    // INPUT(x) / OUTPUT(y)
    const io = trimmedLine.match(/^(INPUT|OUTPUT)\((\w+)\)$/);
    if (io) {
      nodes.push({ 
        id: io[2], 
        type: io[1], 
        inputs: [], 
        size,
        group: io[1].toUpperCase() // Group by INPUT/OUTPUT
      });
      defined.add(io[2]);
      continue;
    }

    // n1 = AND(a, b, c)  # size=0.97
    const gate = trimmedLine.match(/^(\w+)\s*=\s*(\w+)\(([\w,\s]+)\)/);
    if (gate) {
      const [, id, typ, ins] = gate;
      const inputs = ins.split(/\s*,\s*/);
      // Group nodes by their gate type
      nodes.push({ 
        id, 
        type: typ, 
        inputs, 
        size,
        group: typ.toUpperCase() 
      });
      defined.add(id);
      inputs.forEach(i => referenced.add(i));
    }
  }
  
  // Add dummy INPUT nodes for any referenced but not defined
  for (const ref of referenced) {
    if (!defined.has(ref)) {
      nodes.push({ id: ref, type: "INPUT", inputs: [], group: "INPUT" });
      defined.add(ref);
    }
  }

  return nodes;
}

interface CircuitViewProps {
  content: string;
  coordinates?: [number, number][];
  gate_names?: string[];
  maxNodes?: number;
}

function getNodeColor(type: string, id: string) {
  // Color scheme that's easier to distinguish
  switch (type.toUpperCase()) {
    case 'INPUT': return '#818cf8'; // Indigo (was light blue)
    case 'OUTPUT': return '#fb7185'; // Pink/red
    case 'AND': return '#a78bfa'; // Purple
    case 'NAND': return '#8b5cf6'; // Deeper purple
    case 'OR': return '#fb923c'; // Orange
    case 'NOR': return '#f97316'; // Deeper orange
    case 'NOT': return '#c084fc'; // Purple
    case 'BUFF': return '#facc15'; // Yellow
    case 'XOR': return '#4ade80'; // Green
    case 'XNOR': return '#22c55e'; // Deeper green
    default: return '#94a3b8'; // Slate gray
  }
}

function getNodeShape(type: string) {
  if (type.toUpperCase() === 'INPUT' || type.toUpperCase() === 'OUTPUT') return 'ellipse';
  return 'rectangle';
}

function CustomNode(props: NodeProps) {
  const { data, id } = props;
  const color = getNodeColor(data.label, id);
  const shape = getNodeShape(data.label);
  const style: React.CSSProperties = {
    width: 100,
    height: 36,
    background: color,
    border: '1px solid #1e293b',
    borderRadius: shape === 'ellipse' ? 18 : 6,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
    boxSizing: 'border-box',
    textAlign: 'center',
    position: 'relative',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  };
  
  return (
    <div style={style} title={`${id} (${data.label})`}>
      {/* Add source handle on top */}
      <Handle
        type="source"
        position={Position.Top}
        id="source"
        style={{ background: '#555', width: 10, height: 10 }}
      />
      
      {/* Display the node label */}
      <div>
        <div>{data.nodeId}</div>
        <div style={{ fontSize: '9px', opacity: 0.8 }}>{data.label}</div>
      </div>
      
      {/* Add target handle on bottom */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="target"
        style={{ background: '#555', width: 10, height: 10 }}
      />
    </div>
  );
}

const nodeTypes = {
  custom: CustomNode
};

// Search component to find nodes
function NodeSearch({ nodes, onSelectNode }: { nodes: Node[], onSelectNode: (nodeId: string) => void }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState<Node[]>([]);

  useEffect(() => {
    if (!searchTerm) {
      setResults([]);
      return;
    }
    
    const filtered = nodes.filter(node => 
      node.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
      node.data.label.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    // Limit results to avoid overwhelming the UI
    setResults(filtered.slice(0, 15));
  }, [searchTerm, nodes]);

  return (
    <div style={{ background: 'white', padding: '10px', borderRadius: '6px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search nodes..."
        style={{ 
          width: '100%', 
          padding: '8px', 
          borderRadius: '4px', 
          border: '1px solid #ddd',
          marginBottom: '8px'
        }}
      />
      {results.length > 0 && (
        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
          {results.map(node => (
            <div
              key={node.id}
              onClick={() => {
                onSelectNode(node.id);
                setSearchTerm('');
              }}
              style={{
                padding: '6px',
                cursor: 'pointer',
                borderRadius: '4px',
                marginBottom: '2px',
                borderLeft: `4px solid ${getNodeColor(node.data.label, node.id)}`,
                backgroundColor: '#f8fafc',
              }}
            >
              <div><strong>{node.id}</strong></div>
              <div style={{ fontSize: '11px', opacity: 0.8 }}>{node.data.label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function createSimplifiedGraph(data: CircuitNode[], maxNodes: number): CircuitNode[] {
  // If data is smaller than maxNodes, return as is
  if (data.length <= maxNodes) return data;
  
  // Focus on input, output, and important intermediate nodes
  const inputs = data.filter(n => n.type.toUpperCase() === 'INPUT');
  const outputs = data.filter(n => n.type.toUpperCase() === 'OUTPUT');
  
  // Limit the number of inputs and outputs if there are too many
  const maxInputs = Math.floor(maxNodes * 0.25);
  const maxOutputs = Math.floor(maxNodes * 0.25);
  
  const selectedInputs = inputs.length <= maxInputs 
    ? inputs 
    : inputs.slice(0, maxInputs);
    
  const selectedOutputs = outputs.length <= maxOutputs 
    ? outputs 
    : outputs.slice(0, maxOutputs);
  
  // Extract logical groups
  const intermediatesByType = new Map<string, CircuitNode[]>();
  
  // Group by gate type
  data.forEach(node => {
    if (node.type.toUpperCase() !== 'INPUT' && node.type.toUpperCase() !== 'OUTPUT') {
      const type = node.type.toUpperCase();
      if (!intermediatesByType.has(type)) {
        intermediatesByType.set(type, []);
      }
      intermediatesByType.get(type)!.push(node);
    }
  });
  
  // Select nodes based on connections to inputs/outputs and gate types
  const importantNodeIds = new Set<string>();
  
  // Add output nodes and their direct inputs
  selectedOutputs.forEach(out => {
    importantNodeIds.add(out.id);
    
    // Try to find direct inputs to outputs
    const directInputs = data.filter(n => n.inputs.includes(out.id));
    directInputs.slice(0, 5).forEach(n => importantNodeIds.add(n.id)); // Take up to 5 inputs per output
  });
  
  // Add input nodes
  selectedInputs.forEach(input => {
    importantNodeIds.add(input.id);
    
    // Try to find direct outputs from inputs
    const directOutputs = data.filter(n => n.inputs.includes(input.id));
    directOutputs.slice(0, 5).forEach(n => importantNodeIds.add(n.id)); // Take up to 5 outputs per input
  });
  
  // Distribute remaining slots among gate types
  const remainingSlots = maxNodes - importantNodeIds.size;
  if (remainingSlots > 0 && intermediatesByType.size > 0) {
    const slotsPerType = Math.max(3, Math.floor(remainingSlots / intermediatesByType.size));
    
    intermediatesByType.forEach((nodes, type) => {
      // Get samples evenly throughout the array for better distribution
      const step = Math.max(1, Math.floor(nodes.length / slotsPerType));
      for (let i = 0; i < nodes.length && importantNodeIds.size < maxNodes; i += step) {
        importantNodeIds.add(nodes[i].id);
      }
    });
  }
  
  // Filter original data to only include important nodes
  return data.filter(node => importantNodeIds.has(node.id));
}

function improveLayout(nodes: Node[], edges: Edge[], isHorizontal = false): { nodes: Node[], edges: Edge[] } {
  if (nodes.length === 0) return { nodes, edges };

  const g = new dagre.graphlib.Graph();
  g.setGraph({ 
    rankdir: isHorizontal ? "LR" : "TB", 
    nodesep: 50, 
    ranksep: 100,
    ranker: "network-simplex",
    align: "UL"
  });
  g.setDefaultEdgeLabel(() => ({}));
  
  // Add nodes to graph
  nodes.forEach(node => {
    g.setNode(node.id, { width: 100, height: 36 });
  });
  
  // Add edges to graph
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target);
  });
  
  // Apply layout
  dagre.layout(g);
  
  // Get nodes with new positions
  const layoutedNodes = nodes.map(node => {
    const nodeWithPosition = g.node(node.id);
    
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 50, // Adjust for node width/2
        y: nodeWithPosition.y - 18, // Adjust for node height/2
      }
    };
  });

  return { nodes: layoutedNodes, edges };
}

function useBenchGraph(content: string, coordinates?: [number, number][], gate_names?: string[], maxNodes = 800) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    try {
      const allData = parseBench(content);
      
      // Create a simplified version if the circuit is too large
      const data = createSimplifiedGraph(allData, maxNodes);
      
      // Create nodes and edges
      const initialNodes = data.map(circuitNode => ({
        id: circuitNode.id,
        data: { 
          label: circuitNode.type,
          nodeId: circuitNode.id,
          group: circuitNode.group || circuitNode.type
        },
        position: { x: 0, y: 0 }, // Temporary position, will be updated by layout
        type: 'custom',
      } as Node));
      
      // Only create edges where both source and target exist in our simplified graph
      const nodeIds = new Set(data.map(n => n.id));
      
      // Use a Set to track edge IDs and prevent duplicates
      const edgeIdSet = new Set<string>();
      
      const initialEdges: Edge[] = [];
      
      // Create edges with unique IDs
      data.forEach(n => {
        n.inputs.forEach(src => {
          if (nodeIds.has(src)) {
            // Create a base ID for the edge
            const baseEdgeId = `${src}-${n.id}`;
            
            // Make sure the edge ID is unique
            let uniqueEdgeId = baseEdgeId;
            let counter = 1;
            
            // If this edge ID already exists, append a counter to make it unique
            while (edgeIdSet.has(uniqueEdgeId)) {
              uniqueEdgeId = `${baseEdgeId}-${counter}`;
              counter++;
            }
            
            // Add the unique ID to our tracking set
            edgeIdSet.add(uniqueEdgeId);
            
            // Create the edge with the unique ID
            initialEdges.push({
              id: uniqueEdgeId,
              source: src,
              target: n.id,
              sourceHandle: 'source',
              targetHandle: 'target',
              animated: false,
              style: { stroke: '#94a3b8', strokeWidth: 1.5 }
            });
          }
        });
      });
      
      // Improve the layout
      const { nodes: layoutedNodes, edges: layoutedEdges } = improveLayout(
        initialNodes,
        initialEdges,
        data.length < 100 // Use horizontal layout for small circuits
      );
      
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
    } catch (error) {
      console.error("Error processing circuit:", error);
      setNodes([]);
      setEdges([]);
    }
  }, [content, coordinates, gate_names, maxNodes]);

  return { nodes, edges };
}

// The internal CircuitView component that uses the ReactFlow hooks
function CircuitViewComponent({ content, coordinates, gate_names, maxNodes = 800 }: CircuitViewProps) {
  const { nodes, edges } = useBenchGraph(content, coordinates, gate_names, maxNodes);
  const [zoomLevel, setZoomLevel] = useState(0.5);
  const [showSearch, setShowSearch] = useState(false);
  const reactFlowInstance = useReactFlow();

  const focusNode = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (node && reactFlowInstance) {
      reactFlowInstance.setCenter(node.position.x, node.position.y, { zoom: 2, duration: 800 });
    }
  }, [nodes, reactFlowInstance]);

  const toggleGroups = useCallback((group: string) => {
    // Future enhancement: toggle visibility of node groups
  }, []);

  return (
    <div className="h-[500px] w-full border rounded-lg">
      <ReactFlow 
        nodes={nodes} 
        edges={edges} 
        fitView
        minZoom={0.1}
        maxZoom={3}
        defaultViewport={{ x: 0, y: 0, zoom: zoomLevel }}
        className="bg-gray-50"
        nodeTypes={nodeTypes}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background color="#aaa" gap={16} />
        <Controls showInteractive={true} />
        
        <Panel position="top-right">
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={() => setShowSearch(!showSearch)}
              style={{
                background: '#1e293b',
                color: 'white',
                padding: '8px 16px',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer',
                border: 'none'
              }}
            >
              {showSearch ? 'Close Search' : 'Search Nodes'}
            </button>
            <button 
              onClick={() => reactFlowInstance?.fitView({ padding: 0.2, duration: 800 })}
              style={{
                background: '#1e293b',
                color: 'white',
                padding: '8px 16px',
                borderRadius: '4px',
                fontSize: '12px',
                cursor: 'pointer',
                border: 'none'
              }}
            >
              Fit View
            </button>
          </div>
          {showSearch && (
            <div style={{ marginTop: '10px' }}>
              <NodeSearch nodes={nodes} onSelectNode={focusNode} />
            </div>
          )}
        </Panel>
      </ReactFlow>
    </div>
  );
}

// The main CircuitView component that wraps with ReactFlowProvider
export default function CircuitView(props: CircuitViewProps) {
  return (
    <ReactFlowProvider>
      <CircuitViewComponent {...props} />
    </ReactFlowProvider>
  );
} 