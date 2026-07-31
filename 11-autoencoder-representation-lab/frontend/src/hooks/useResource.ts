import {useEffect, useState} from "react";

export function useResource<T>(loader:()=>Promise<T>, dependencies:unknown[] = []) {
  const [data,setData] = useState<T|null>(null);
  const [error,setError] = useState("");
  const [loading,setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    setLoading(true);
    loader().then(value => {
      if (active) { setData(value); setError(""); }
    }).catch(reason => {
      if (active) setError(reason instanceof Error ? reason.message : "Request failed");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, dependencies);
  return {data,error,loading};
}
