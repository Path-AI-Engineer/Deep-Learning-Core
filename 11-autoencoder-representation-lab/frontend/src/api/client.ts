const API = import.meta.env.VITE_API_URL ?? "/api/v1";

async function request<T>(path:string, init?:RequestInit):Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({detail:"Request failed"}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  get:<T>(path:string) => request<T>(path),
  post:<T>(path:string, body:unknown) => request<T>(path, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body),
  }),
  upload:<T>(path:string, file:File) => request<T>(path, {
    method:"POST",
    headers:{"Content-Type":file.type},
    body:file,
  }),
};
