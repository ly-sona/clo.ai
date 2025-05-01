import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import UploadDropzone from "../components/UploadDropzone";
import LayoutCard from "../components/LayoutCard";
import { SimulationModal } from "../components/SimulationModal";
import { apiGetJson } from "../api/client";

/** ------------------------------------------------------------------
 *  Types
 * ------------------------------------------------------------------*/
export type LayoutMeta = { id: string; thumb: string; power: number };

/** ------------------------------------------------------------------
 *  Shared background component for consistent styling across pages
 * ------------------------------------------------------------------*/
export function BackgroundPattern() {
  return (
    <div className="absolute inset-0">
      <div className="relative h-full w-full bg-slate-950 [&>div]:absolute [&>div]:inset-0 [&>div]:bg-[radial-gradient(circle_500px_at_50%_200px,#3e3e3e,transparent)]">
        <div></div>
      </div>
    </div>
  );
}

/** ------------------------------------------------------------------
 *  Navbar with consistent styling
 * ------------------------------------------------------------------*/
export function NavBar() {
  return (
    <header className="relative z-10">
      <div className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto text-white">
        <div className="flex items-center gap-2 text-xl font-semibold">
          <span className="inline-block h-3 w-3 rounded-full bg-sky-400" />
          <span>clo.ai</span>
        </div>
        <nav className="flex items-center gap-8 text-sm">
          <a href="/" className="hover:text-sky-400 transition-colors">Home</a>
          <a href="/about" className="hover:text-sky-400 transition-colors">About</a>
          <a href="https://github.com" className="hover:text-sky-400 transition-colors">GitHub</a>
          <a href="#" className="rounded-lg border px-6 py-3 font-medium border-slate-700 bg-slate-800 text-white hover:bg-slate-700 transition-colors">
            Contact
          </a>
        </nav>
      </div>
    </header>
  );
}

/** ------------------------------------------------------------------
 *  Main page
 * ------------------------------------------------------------------*/
export default function Home() {
  const [layouts, setLayouts] = useState<LayoutMeta[]>([]);
  const [sim, setSim] = useState<string | null>(null);

  const refresh = async () => {
    const data = await apiGetJson<LayoutMeta[]>("/layouts");
    setLayouts(data);
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    const job = new URLSearchParams(location.search).get("job");
    if (job) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <NavBar />

      {/* Hero ---------------------------------------------------- */}
      <section className="relative z-10 flex min-h-[80vh] flex-col items-center justify-center px-4">
        <div className="max-w-3xl text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8 text-4xl font-bold tracking-tight sm:text-6xl lg:text-7xl text-white"
          >
            Chip Layout <span className="text-sky-400">Optimizer</span>
          </motion.h1>
          <p className="mx-auto mb-8 max-w-2xl text-lg text-slate-300">
            Upload your macro-cell schematic and let our evolutionary engine craft
            a power-efficient floorplan in seconds.
          </p>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mx-auto max-w-lg"
          >
            <UploadDropzone onFinished={refresh} />
          </motion.div>
        </div>
      </section>

      {/* Gallery ------------------------------------------------- */}
      {layouts.length > 0 && (
        <section id="gallery" className="relative z-10 py-24 px-4 max-w-7xl mx-auto">
          <h2 className="text-3xl font-semibold text-white mb-8 text-center">
            Latest Layouts
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {layouts.map((l) => (
              <LayoutCard key={l.id} {...l} onSimulate={setSim} />
            ))}
          </div>
        </section>
      )}

      {sim && (
        <SimulationModal
          id={sim}
          open={!!sim}
          onOpenChange={() => setSim(null)}
        />
      )}
    </div>
  );
}
