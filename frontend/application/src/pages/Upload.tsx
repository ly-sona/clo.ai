import { useState } from "react";
import { motion } from "framer-motion";
import UploadDropzone from "../components/UploadDropzone";
import { BackgroundPattern, LayoutMeta } from "./Home";
import LayoutCard from "../components/LayoutCard";
import { RefreshCw } from "lucide-react";

export default function Upload() {
  const [newLayout, setNewLayout] = useState<LayoutMeta | null>(null);
  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <section className="relative z-10 flex min-h-[80vh] flex-col items-center justify-center px-4 pt-8 w-full">
        <div className="w-full max-w-[90%] lg:max-w-[1400px] text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-4 text-4xl font-bold tracking-tight sm:text-6xl lg:text-7xl text-white quantico-bold"
          >
            Upload <span className="text-indigo-400">Circuit</span>
          </motion.h1>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mx-auto w-full flex justify-center"
          >
            {!newLayout ? (
              <UploadDropzone onFinished={layout => setNewLayout(layout ?? null)} />
            ) : null}
          </motion.div>
          {newLayout && (
            <>
              <div className="w-full mx-auto flex flex-col">
                <div className="w-full flex justify-start mb-2">
                  <button
                    className="animated-gradient px-6 py-3 rounded-lg bg-gradient-to-r from-indigo-300 via-sky-200 to-purple-300 text-slate-900 font-semibold shadow flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-sky-300/50"
                    onClick={() => setNewLayout(null)}
                    title="Upload Again"
                  >
                    <RefreshCw className="w-5 h-5" />
                  </button>
                </div>
                <LayoutCard {...newLayout} />
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
} 