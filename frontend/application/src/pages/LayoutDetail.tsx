import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import LayoutCard from "../components/LayoutCard";
import { apiGetJson } from "../api/client";
import { BackgroundPattern, LayoutMeta } from "./Home";

export default function LayoutDetail() {
  const { id } = useParams();
  const [layout, setLayout] = useState<LayoutMeta | null>(null);

  useEffect(() => {
    if (id) {
      apiGetJson<LayoutMeta>(`/layout/${id}`).then(setLayout);
    }
  }, [id]);

  return (
    <div className="relative min-h-screen">
      <BackgroundPattern />
      <div className="flex justify-center items-center min-h-[60vh] pt-6">
        {layout ? (
          <LayoutCard {...layout} />
        ) : (
          <div className="text-white">Loading...</div>
        )}
      </div>
    </div>
  );
} 