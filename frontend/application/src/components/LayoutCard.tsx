import { API } from "../api/client";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader } from "./ui/card";

interface Props {
  id: string;
  thumb: string;   // now matches backend
  power: number;
  onSimulate: (id: string) => void;
}

export default function LayoutCard({ id, thumb, power, onSimulate }: Props) {
  return (
    <Card className="hover:shadow-xl transition-shadow duration-200">
      <CardHeader>
        <h3 className="text-lg font-semibold">Layout&nbsp;{id.slice(0, 8)}</h3>
      </CardHeader>

      <CardContent className="flex flex-col items-center gap-4">
        {/* prepend API so it works in dev & prod */}
        <img
          src={`${API}${thumb}`}
          alt={`layout ${id}`}
          className="w-full rounded-xl"
        />

        <p className="text-sm text-muted-foreground">
          Power:&nbsp;{power.toFixed(3)}&nbsp;mW
        </p>

        <Button onClick={() => onSimulate(id)} className="w-full">
          Simulate
        </Button>
      </CardContent>
    </Card>
  );
}