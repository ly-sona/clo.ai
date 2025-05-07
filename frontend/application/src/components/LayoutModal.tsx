import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "./ui/dialog";
import LayoutCard from "./LayoutCard";
import type { LayoutMeta } from "../pages/Home";
import { X } from "lucide-react";

interface LayoutModalProps {
  layout: LayoutMeta;
  original_power?: number;
  original_delay?: number;
  new_delay?: number;
  trigger: React.ReactNode;
}

export default function LayoutModal({ layout, original_power, original_delay, new_delay, trigger }: LayoutModalProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        {trigger}
      </DialogTrigger>
      <DialogContent 
        className="w-[50vw] max-w-[50vw] h-[95vh] bg-white/95 backdrop-blur-md border border-slate-200 shadow-2xl rounded-xl p-0 overflow-hidden"
        hideDefaultCloseButton
      >
        {/* Custom header with gradient */}
        <div className="relative border-b border-slate-200 bg-gradient-to-r from-indigo-50 via-white to-indigo-100 p-5">
          <DialogHeader className="pr-10">
            <DialogTitle className="text-xl quantico-bold text-slate-800 flex items-center">
              Layout Details: <span className="text-indigo-600 ml-2">{layout.id}</span>
            </DialogTitle>
          </DialogHeader>
          
          {/* Custom close button */}
          <DialogClose className="absolute right-4 top-4 rounded-full p-1.5 text-slate-500 hover:text-indigo-700 hover:bg-indigo-100 transition-colors">
            <X className="h-5 w-5" />
            <span className="sr-only">Close</span>
          </DialogClose>
        </div>
        
        {/* Content area with nested background pattern */}
        <div className="p-6 pt-8 relative overflow-y-auto h-[calc(95vh-80px)]">
          {/* Background pattern */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_800px_at_100%_200px,rgba(99,102,241,0.08),transparent)]"></div>
          
          {/* Actual content */}
          <div className="relative">
            <LayoutCard 
              {...layout} 
              original_power={original_power}
              original_delay={original_delay}
              new_delay={new_delay}
              isInModal={true}
            />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
} 