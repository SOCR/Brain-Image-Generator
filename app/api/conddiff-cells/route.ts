import { NextResponse } from "next/server";
import { callSpace } from "@/lib/hfSpace";

/**
 * GET /api/conddiff-cells
 *
 * Proxies the Python backend's /conddiff-cells, which reports the (Lobe, Slice Location)
 * combinations braingen_CondDiffuser_BraTS_v1 can actually generate.
 *
 * WHY THIS ROUTE EXISTS AT ALL, given NEXT_PUBLIC_BACKEND_URL is readable by the browser:
 * every other backend call in this app is proxied through app/api/ (see generate/route.ts), so
 * a direct browser fetch would be the only client-side cross-origin call in the codebase - it
 * would depend on api.py's CORS allowed_origins list staying in sync with whatever Vercel
 * preview URL is deployed, and it would fail silently on a preview domain nobody added there.
 * Proxying keeps the request same-origin and makes the backend URL a server-side concern.
 *
 * WHY THE FRONTEND NEEDS IT: the conditioning bank fills only 11 of the 18 lobe x level cells
 * (the cerebellum has no superior slices; the insula may never clear the lobe-area gate). If
 * the UI offered all 18, a user could pick a cell that does not exist, and the backend's
 * blanket exception handler would turn that into HTTP 200 with no images - a "Success" toast
 * over an empty viewer. The panel greys the missing cells out instead.
 *
 * NO AUTH CHECK, deliberately, unlike generate/route.ts. This returns nothing but the model's
 * own capability matrix - no user data, no generation, no cost. The playground renders the
 * parameter panel before a session exists, so requiring auth here would leave every lobe
 * disabled for exactly the users most likely to be trying the model out.
 */
export async function GET() {
  try {
    // Hugging Face Space backend: same JSON. A throw (asleep, unreachable) falls into the catch
    // below and reports ready:false, exactly like an unreachable FastAPI backend.
    if (process.env.HF_SPACE_URL) {
      return NextResponse.json(await callSpace(process.env.HF_SPACE_URL, "conddiff_cells", [], process.env.HF_TOKEN));
    }

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

    const response = await fetch(`${backendUrl}/conddiff-cells`, {
      // The bank changes only when someone re-exports and redeploys it, but caching a
      // "ready: false" from a backend that was still booting would leave the controls disabled
      // for the lifetime of the cache. Always ask.
      cache: "no-store",
    });

    if (!response.ok) {
      // A backend that is down or too old to know this route is NOT an error the user should
      // see. Report unready with empty lists and let the panel decide how to present that -
      // the same shape the backend itself returns when it is not READY, so the client has one
      // code path instead of two.
      return NextResponse.json(
        { mode: "unknown", ready: false, reason: `backend returned ${response.status}`,
          pairs: [], lobes: [], levels: [], sizes: [] },
        { status: 200 }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error("Error fetching conddiff cells:", error);
    // Say WHICH backend was tried and why it failed. "backend unreachable" alone made a missing
    // HF_SPACE_URL (fallback to the old FastAPI URL) look identical to a failing Space call.
    // The message never contains the token: callSpace errors carry only an HTTP status or text.
    const tried = process.env.HF_SPACE_URL ? "HF Space" : "FastAPI fallback (HF_SPACE_URL not set)";
    const why = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { mode: "unknown", ready: false, reason: `backend unreachable: ${tried}: ${why}`,
        pairs: [], lobes: [], levels: [], sizes: [] },
      { status: 200 }
    );
  }
}
