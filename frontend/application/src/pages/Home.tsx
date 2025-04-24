// src/pages/Home.tsx
import { useEffect, useState } from "react";
import UploadDropzone from "../components/UploadDropzone";
import LayoutCard from "../components/LayoutCard";
import { SimulationModal } from "../components/SimulationModal";
import { apiGetJson } from "../api/client";

export default function Home() {
  const [layouts, setLayouts] = useState<any[]>([]);
  const [sim, setSim] = useState<string|null>(null);

  useEffect(() => {
    apiGetJson("/layouts").then(setLayouts);   // returns [{id, thumb, power}]
  }, []);

  return (
    <div className="py-10 px-4 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8 text-center">Chip Layout Optimizer</h1>
      <UploadDropzone />

      {/* Gallery */}
      <div className="grid md:grid-cols-3 gap-6 mt-12">
        {layouts.map(l => (
          <LayoutCard key={l.id} {...l} onSimulate={setSim} />
        ))}
      </div>

      {sim && <SimulationModal id={sim} open={!!sim} onOpenChange={()=>setSim(null)} />}
    </div>
  );
}