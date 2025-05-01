import { motion } from "framer-motion";
import { BackgroundPattern, NavBar } from "./Home";

export default function About() {
  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <NavBar />
      
      <section className="relative z-10 flex min-h-[80vh] flex-col items-center justify-center px-4">
        <div className="max-w-3xl text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8 text-4xl font-bold tracking-tight sm:text-6xl lg:text-7xl text-white"
          >
            About <span className="text-sky-400">clo.ai</span>
          </motion.h1>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <p className="mx-auto mb-12 max-w-2xl text-lg text-slate-300">
              clo.ai is an advanced chip layout optimization platform that leverages evolutionary algorithms 
              to create efficient floorplans for integrated circuits. Our platform helps hardware designers 
              optimize power consumption and thermal distribution while maintaining performance requirements.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="rounded-lg border p-6 border-slate-700 bg-slate-800/50">
                <h3 className="text-xl font-semibold mb-4 text-white">Our Mission</h3>
                <p className="text-slate-300">
                  To revolutionize chip design by making advanced layout optimization 
                  accessible to hardware engineers worldwide.
                </p>
              </div>
              
              <div className="rounded-lg border p-6 border-slate-700 bg-slate-800/50">
                <h3 className="text-xl font-semibold mb-4 text-white">Technology</h3>
                <p className="text-slate-300">
                  Powered by state-of-the-art evolutionary algorithms and thermal modeling 
                  to deliver optimal chip layouts in seconds.
                </p>
              </div>
            </div>

            <div className="mt-12 flex flex-wrap justify-center gap-4">
              <button className="rounded-lg px-6 py-3 font-medium bg-sky-400 text-slate-900 hover:bg-sky-300">
                Get Started
              </button>
              <button className="rounded-lg border px-6 py-3 font-medium border-slate-700 bg-slate-800 text-white hover:bg-slate-700">
                Learn More
              </button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}