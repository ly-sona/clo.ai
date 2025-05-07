import { useEffect, useState } from "react";
import LayoutCardSimple from "../components/LayoutCardSimple";
import { apiGetJson } from "../api/client";
import { BackgroundPattern, LayoutMeta } from "./Home";

export default function Layouts() {
  const [layouts, setLayouts] = useState<LayoutMeta[]>([]);

  useEffect(() => {
    apiGetJson<LayoutMeta[]>("/layouts").then(setLayouts);
  }, []);

  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <section className="relative z-10 py-12 px-4 w-full max-w-[90%] lg:max-w-[1400px] mx-auto">
        <div>
          <h2 className="text-3xl font-semibold text-white mb-6 text-center quantico-bold">
            Layouts
          </h2>
          {layouts.length === 0 ? (
            <div className="text-center text-white">No layouts yet. Upload a circuit to get started!</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {layouts.map((l) => (
                <LayoutCardSimple key={l.id} {...l} />
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}