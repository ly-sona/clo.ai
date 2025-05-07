import { motion } from "framer-motion";

/** ------------------------------------------------------------------
 *  Types
 * ------------------------------------------------------------------*/
export type LayoutMeta = { 
  id: string; 
  thumb: string; 
  power: number;
  optimized_circuit?: string;
  coordinates?: [number, number][];
  gate_names?: string[];
};

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
    <header className="relative z-10 w-full backdrop-blur-sm bg-slate-950/60 border-b border-slate-800/50">
      <div className="flex items-center justify-between px-6 py-4 max-w-[90%] lg:max-w-[1400px] mx-auto text-white">
        <a href="/" className="flex items-center gap-2 text-xl font-semibold quantico-bold hover:text-indigo-400 transition-colors">
          <span className="inline-block h-3 w-3 rounded-full bg-gradient-to-r from-indigo-400 to-purple-400 logo-dot"></span>
          <span>clo.ai</span>
        </a>
        <nav className="flex items-center gap-8 text-sm">
          <a href="/layouts" className="hover:text-indigo-400 transition-colors">Layouts</a>
          <a href="https://github.com/ly-sona/clo.ai" className="hover:text-indigo-400 transition-colors">GitHub</a>
          <a href="/upload" className="animated-gray-gradient rounded-lg px-6 py-3 font-medium bg-gradient-to-r from-slate-700 via-slate-600 to-slate-800 text-white shadow-sm hover:shadow-indigo-900/20 hover:shadow-md transition-all">
            Upload
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
  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />

      {/* Hero ---------------------------------------------------- */}
      <section className="relative z-10 flex min-h-[90vh] flex-col items-center justify-center px-4 pt-0">
        <div className="w-full max-w-[90%] lg:max-w-[1400px] text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-4 text-4xl font-bold tracking-tight sm:text-6xl lg:text-7xl text-white quantico-bold"
          >
            clo<span className="text-indigo-400">.ai</span>
          </motion.h1>
          <p className="mx-auto mb-4 max-w-2xl text-lg text-slate-300 quantico-regular">
            XGBoost + DEAP Based VLSI Layout Optimizer
          </p>

          <a
            href="/upload"
            className="animated-gradient inline-block px-8 py-4 rounded-xl text-lg font-semibold bg-gradient-to-r from-indigo-300 via-sky-200 to-purple-300 text-slate-900 shadow-lg focus:outline-none focus:ring-4 focus:ring-indigo-300/50"
          >
            <span className="relative z-10">Get Started &rarr;</span>
          </a>
        </div>
      </section>

      {/* Spacer div to ensure content is below the viewport */}
      <div className="h-[15vh]"></div>

      {/* Why Faster = Greener Section ----------------------------- */}
      <section className="relative z-10 py-10 px-4 mb-0">
        <div className="w-full max-w-[90%] lg:max-w-[1400px] mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            viewport={{ once: true, amount: 0.3 }}
            className="mb-0"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-12 flex items-center">
              <span className="text-indigo-400 mr-4">⚡</span> Why Faster = Greener
            </h2>
            <p className="text-slate-300 mb-8 max-w-3xl">
              Delay is the time it takes a signal to travel from the input of a circuit to its output. 
              Trimming that time helps power in three complementary ways:
            </p>

            <div className="grid md:grid-cols-3 gap-8 mt-12">
              <div className="bg-slate-900/60 border border-slate-800/70 p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-indigo-300 mb-4">Lower Switching Activity</h3>
                <p className="text-slate-300">
                  Every extra picosecond on the critical path keeps nodes in an indeterminate state, 
                  letting glitches ripple through downstream logic. Fewer glitches → fewer unnecessary 
                  transitions → less dynamic power ( P ∝ C V² f ).
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/70 p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-indigo-300 mb-4">Slack Turns Into Voltage‑Headroom</h3>
                <p className="text-slate-300">
                  When the path is shorter, you can hit the same clock target at a lower supply voltage 
                  or reclaim head‑room for extra functionality. Because power scales quadratically with V, 
                  even a small voltage drop yields outsized savings.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/70 p-6 rounded-xl">
                <h3 className="text-xl font-semibold text-indigo-300 mb-4">Smaller, Cooler Gates</h3>
                <p className="text-slate-300">
                  Many delay fixes come from right‑sizing or relocating gates. Smaller transistors bring 
                  down both capacitive loading and leakage. Cooler chips in turn need less guard‑banding, 
                  creating a virtuous circle.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* How the Layout-Optimization Model Works ------------------- */}
      <section className="relative z-10 py-2 px-4 pb-16 bg-slate-950/70 mt-0 mb-0">
        <div className="w-full max-w-[90%] lg:max-w-[1400px] mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="w-full pb-8"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-12 flex items-center justify-end">
              <span className="text-indigo-400 ml-4 order-2">🛠️</span>
              <span className="order-1">How the Layout‑Optimization Model Works</span>
            </h2>

            <div className="overflow-x-auto">
              <table className="min-w-full bg-transparent">
                <thead>
                  <tr>
                    <th className="py-4 px-6 bg-slate-800/60 text-left text-sm font-semibold text-white tracking-wider rounded-tl-lg">
                      Stage
                    </th>
                    <th className="py-4 px-6 bg-slate-800/60 text-left text-sm font-semibold text-white tracking-wider rounded-tr-lg">
                      What Happens
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  <tr className="bg-slate-900/40">
                    <td className="py-4 px-6 text-sm font-medium text-indigo-300">1. Parse & Sketch</td>
                    <td className="py-4 px-6 text-sm text-slate-300">
                      We read ISCAS‑85 or CircuitNet benches, discover inputs → gates → outputs, and auto‑assign (x, y) coordinates in a leveled "logic skyline."
                    </td>
                  </tr>
                  <tr className="bg-slate-900/30">
                    <td className="py-4 px-6 text-sm font-medium text-indigo-300">2. Simulate & Sample</td>
                    <td className="py-4 px-6 text-sm text-slate-300">
                      A quick transistor‑sizing sweep gives us thousands of &lt;del, power&gt; pairs. These become the training labels.
                    </td>
                  </tr>
                  <tr className="bg-slate-900/40">
                    <td className="py-4 px-6 text-sm font-medium text-indigo-300">3. Learn the Physics</td>
                    <td className="py-4 px-6 text-sm text-slate-300">
                      An XGBoost regressor learns the non‑linear dance between gate count, fan‑in, placement density and delay/power. Think of it as a digital wind‑tunnel.
                    </td>
                  </tr>
                  <tr className="bg-slate-900/30">
                    <td className="py-4 px-6 text-sm font-medium text-indigo-300">4. Search for Better Layouts</td>
                    <td className="py-4 px-6 text-sm text-slate-300">
                      A genetic algorithm (DEAP) mutates one global "sizing weight," feeds it through the model, and keeps whichever chromosome cuts predicted RMSE on delay.
                    </td>
                  </tr>
                  <tr className="bg-slate-900/40 rounded-b-lg">
                    <td className="py-4 px-6 text-sm font-medium text-indigo-300 rounded-bl-lg">5. Validate & Export</td>
                    <td className="py-4 px-6 text-sm text-slate-300 rounded-br-lg">
                      The winning weight is written back into the netlist, then re‑simulated for ground‑truth power and timing.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
