/**
 * Client for the Hugging Face ZeroGPU Space backend (backend/hf_space/app.py).
 *
 * Used by the API routes only (server side): callSpace(process.env.HF_SPACE_URL, ..., process.env.HF_TOKEN).
 * ZeroGPU bills GPU time to the CALLER, and the token makes the caller the lab's HF account, so
 * every diffusion image generated through the site draws on that account's daily quota
 * (backend/hf_space/DEPLOY.md §4).
 *
 * WHY NOT JUST fetch() THE SPACE LIKE THE OLD BACKEND: a ZeroGPU Space must be a Gradio app, and
 * a Gradio endpoint is not one POST. It is Gradio's documented two-step HTTP API:
 *   1. POST {space}/gradio_api/call/{endpoint}  {"data": [...args]}  ->  {"event_id": "..."}
 *   2. GET  {space}/gradio_api/call/{endpoint}/{event_id}            ->  a server-sent-event stream
 *      that ends with "event: complete / data: [result]" or "event: error / data: {error: msg}".
 * That is ~20 lines, so it is written out here rather than adding the @gradio/client package.
 *
 * Keep it server-side: the token is a server-only env var (never NEXT_PUBLIC_), because anything
 * browser code has, every visitor can read.
 */
export async function callSpace(base: string, endpoint: string, args: unknown[], token?: string): Promise<any> {
  base = base.replace(/\/+$/, "")
  const headers: Record<string, string> = { "Content-Type": "application/json" }
  if (token) headers.Authorization = `Bearer ${token}`

  // A free Space sleeps after a period without traffic and the first request wakes it. While it
  // boots, HF answers 502/503; an unreachable host makes fetch() itself reject.
  const queued = await fetch(`${base}/gradio_api/call/${endpoint}`, {
    method: "POST",
    headers,
    body: JSON.stringify({ data: args }),
    cache: "no-store",
  }).catch(() => null)
  if (!queued?.ok) {
    throw new Error(
      `The model server is not answering${queued ? ` (HTTP ${queued.status})` : ""}. If it was ` +
      `asleep it is now starting up, which takes a minute or two. Please try again shortly.`
    )
  }
  const { event_id } = await queued.json()

  // The server closes the stream after the final event, so reading it whole is enough. Only the
  // last "event:/data:" block matters; earlier ones are heartbeats.
  const stream = await fetch(`${base}/gradio_api/call/${endpoint}/${event_id}`, { headers, cache: "no-store" })
  const last = (await stream.text()).trim().split(/\r?\n\r?\n/).pop() ?? ""
  const event = /^event: *(.*)$/m.exec(last)?.[1]?.trim()
  const data = /^data: *(.*)$/m.exec(last)?.[1]

  if (event === "complete" && data) return JSON.parse(data)[0]
  // gr.Error messages (including ZeroGPU's "quota exceeded ... try again in H:MM:SS") arrive as
  // the error event's data: {"error": "<message>", "title": ..., ...} in Gradio 6. ZeroGPU
  // sometimes formats them as HTML, and they end up in a toast, so tags are stripped.
  const parsed = data && data !== "null" ? JSON.parse(data) : null
  const message = typeof parsed === "string" ? parsed : parsed?.error
  throw new Error(String(message || `The model server failed on "${endpoint}".`).replace(/<[^>]*>/g, ""))
}
