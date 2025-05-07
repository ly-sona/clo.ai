const API_BASE_URL = 'http://localhost:8000';

export interface OptimizedCircuit {
  job_id: string;
  layouts: Array<{
    id: string;
    thumb: string;
    power: number;
    wns?: number;
    cells?: number;
    fullPng?: string;
  }>;
}

export async function optimizeCircuit(file: File): Promise<OptimizedCircuit> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/optimize`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to optimize circuit');
  }

  return response.json();
}

export async function getLayouts(): Promise<OptimizedCircuit['layouts']> {
  const response = await fetch(`${API_BASE_URL}/layouts`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch layouts');
  }

  return response.json();
}

export async function downloadLayout(layoutId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/layout/${layoutId}/download`);
  
  if (!response.ok) {
    throw new Error('Failed to download layout');
  }

  return response.blob();
} 