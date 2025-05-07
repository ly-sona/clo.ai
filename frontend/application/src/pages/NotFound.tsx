import React from "react";
import { BackgroundPattern } from "./Home";

export default function NotFound() {
  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <div className="flex flex-col justify-center items-center min-h-[80vh] pt-6">
        <h1 className="text-4xl font-bold text-white mb-4 quantico-bold">404</h1>
        <p className="text-xl text-slate-300">Page not found</p>
      </div>
    </div>
  );
}