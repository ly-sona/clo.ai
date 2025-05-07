import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Layouts from "./pages/Layouts";
import Upload from "./pages/Upload";
import NotFound from "./pages/NotFound";
import LayoutDetail from "./pages/LayoutDetail";
import { Toaster } from './components/ui/toaster';
import { NavBar } from "./pages/Home";

export default function App() {
  return (
    <BrowserRouter>
      <div className="relative min-h-screen">
        <NavBar />
        <Routes>
          <Route path="/"            element={<Home />} />
          <Route path="/layouts"     element={<Layouts />} />
          <Route path="/upload"      element={<Upload />} />
          <Route path="/layout/:id"  element={<LayoutDetail />} />
          <Route path="*"            element={<NotFound />} />
        </Routes>
        <Toaster />
      </div>
    </BrowserRouter>
  );
}