export const API = import.meta.env.VITE_API ?? "http://localhost:8000";
export const WS  = API.replace(/^http/, "ws");

export async function apiGetJson<T = unknown>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function apiPostFile(
  path: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<string> {
  const body = new FormData();
  body.append("file", file);

  const xhr = new XMLHttpRequest();
  const url  = `${API}${path}`;

  const p = new Promise<string>((resolve, reject) => {
    xhr.onerror  = () => reject(new Error("upload failed"));
    xhr.onload   = () => {
      try {
        resolve(JSON.parse(xhr.responseText).job_id as string);
      } catch (e) {
        reject(e);
      }
    };
  });

  if (onProgress) {
    xhr.upload.onprogress = (e) => {
      onProgress(Math.round((e.loaded / e.total) * 100));
    };
  }

  xhr.open("POST", url);
  xhr.send(body);
  return p;
}

export async function apiGetLef(): Promise<string> {
  const response = await fetch(`${API}/layout/default/lef`);
  if (!response.ok) {
    throw new Error('Failed to fetch LEF content');
  }
  return response.text();
}

export async function apiGetDef(layoutId: string): Promise<string> {
  const response = await fetch(`${API}/def/${layoutId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch DEF content');
  }
  return response.text();
}
