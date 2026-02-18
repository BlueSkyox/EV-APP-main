import { NextResponse } from "next/server";

type Body = {
  start: { lat: number; lng: number };
  end: { lat: number; lng: number };
};

export async function POST(req: Request) {
  const body = (await req.json()) as Body;

  const ORS_KEY = process.env.ORS_API_KEY;
  if (!ORS_KEY) {
    return NextResponse.json({ error: "Missing ORS_API_KEY" }, { status: 500 });
  }

  // Exemple : appel OpenRouteService (directions)
  const orsRes = await fetch(
    "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
    {
      method: "POST",
      headers: {
        Authorization: ORS_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        coordinates: [
          [body.start.lng, body.start.lat],
          [body.end.lng, body.end.lat],
        ],
      }),
    }
  );

  if (!orsRes.ok) {
    const txt = await orsRes.text();
    return NextResponse.json({ error: "ORS error", details: txt }, { status: 500 });
  }

  const data = await orsRes.json();

  // Tu renverras ici TES métriques (distance, temps, conso, eco-speed, etc.)
  const summary = {
    distance_m: data?.features?.[0]?.properties?.summary?.distance,
    duration_s: data?.features?.[0]?.properties?.summary?.duration,
    geojson: data,
  };

  return NextResponse.json(summary);
}
