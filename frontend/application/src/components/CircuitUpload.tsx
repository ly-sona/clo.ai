import React, { useState } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { useToast } from './ui/use-toast.ts';
import CircuitView from './CircuitView';

interface CircuitUploadProps {
  onOptimize: (file: File) => Promise<void>;
}

export function CircuitUpload({ onOptimize }: CircuitUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.name.endsWith('.bench')) {
        setFile(selectedFile);
        // Read and display the file content
        const reader = new FileReader();
        reader.onload = (e) => {
          setPreview(e.target?.result as string);
        };
        reader.readAsText(selectedFile);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a .bench file",
          variant: "destructive",
        });
      }
    }
  };

  const handleOptimize = async () => {
    if (!file) return;
    
    setIsLoading(true);
    try {
      await onOptimize(file);
      toast({
        title: "Optimization started",
        description: "Your circuit is being optimized. This may take a few minutes.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to start optimization. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="p-6 space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold">Circuit Upload</h2>
        <p className="text-gray-500">Upload your .bench file to optimize the circuit layout</p>
      </div>
      
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <Button
            onClick={handleOptimize}
            disabled={!file || isLoading}
            size="sm"
            className="w-24"
          >
            {isLoading ? "Running..." : "Start"}
          </Button>
          <input
            type="file"
            accept=".bench"
            onChange={handleFileChange}
            className="flex-1"
          />
        </div>

        {preview && (
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Circuit Preview</h3>
            <CircuitView content={preview} />
          </div>
        )}
      </div>
    </Card>
  );
} 