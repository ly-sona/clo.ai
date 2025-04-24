const API = import.meta.env.VITE_API_ROOT ?? "http://localhost:8000";

export async function apiPostFile(
  path: string,
  file: File,
  onProgress?: (pct:number)=>void
) {
  const data = new FormData();
  data.append("file", file);

  const req = new XMLHttpRequest();
  return new Promise<string>((res, rej)=>{
    req.open("POST", `${API}${path}`);
    req.upload.onprogress = e => onProgress?.((e.loaded/e.total)*100);
    req.onreadystatechange = () => {
      if (req.readyState === 4) {
        if (req.status < 400) res(JSON.parse(req.responseText).job_id);
        else rej(req.responseText);
      }
    };
    req.send(data);
  });
}

export const apiGetJson = (path:string)=>fetch(`${API}${path}`).then(r=>r.json());