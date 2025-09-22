// rss_to_csv.js — récupération continue des news BTC depuis CryptoPanic
// Gère la déduplication persistante et normalise les timestamps en Europe/Paris.

import fs from "fs";
import Parser from "rss-parser";

const parser = new Parser();

const FEED_URL = "https://cryptopanic.com/news/rss/?currencies=BTC";
const OUT_FILE = "live_raw.csv";
const SEEN_FILE = "live_seen.json";
const INTERVAL_MS = 15000; // 15 secondes
const MAX_SEEN = 5000;

let seenSet = new Set();
let seenQueue = [];

const parisFormatter = new Intl.DateTimeFormat("sv-SE", {
  timeZone: "Europe/Paris",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

function loadSeen() {
  try {
    if (!fs.existsSync(SEEN_FILE)) return;
    const raw = JSON.parse(fs.readFileSync(SEEN_FILE, "utf8"));
    if (Array.isArray(raw)) {
      raw.forEach((sig) => {
        seenQueue.push(sig);
        seenSet.add(sig);
      });
    }
  } catch (err) {
    console.error("[WARN] Impossible de charger", SEEN_FILE, err.message);
  }
}

function persistSeen() {
  try {
    fs.writeFileSync(SEEN_FILE, JSON.stringify(seenQueue.slice(-MAX_SEEN), null, 2), "utf8");
  } catch (err) {
    console.error("[WARN] Impossible d'écrire", SEEN_FILE, err.message);
  }
}

function remember(sig) {
  if (seenSet.has(sig)) return;
  seenSet.add(sig);
  seenQueue.push(sig);
  if (seenQueue.length > MAX_SEEN) {
    const overflow = seenQueue.splice(0, seenQueue.length - MAX_SEEN);
    overflow.forEach((item) => seenSet.delete(item));
  }
}

function parisParts(date) {
  const parts = parisFormatter
    .formatToParts(date)
    .reduce((acc, part) => ({ ...acc, [part.type]: part.value }), {});
  return {
    year: parts.year,
    month: parts.month,
    day: parts.day,
    hour: parts.hour,
    minute: parts.minute,
    second: parts.second,
  };
}

function parisIso(date) {
  const { year, month, day, hour, minute, second } = parisParts(date);
  const parisMs = Date.UTC(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second)
  );
  const offsetMinutes = Math.round((parisMs - date.getTime()) / 60000);
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const offHour = String(Math.floor(abs / 60)).padStart(2, "0");
  const offMin = String(abs % 60).padStart(2, "0");
  return `${year}-${month}-${day}T${hour}:${minute}:${second}${sign}${offHour}:${offMin}`;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const str = String(value);
  if (str.includes('"') || str.includes(',') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function writeHeaderIfNeeded() {
  if (!fs.existsSync(OUT_FILE)) {
    const header = [
      "datetime_paris",
      "datetime_utc",
      "title",
      "url",
      "summary",
      "source",
      "news_id"
    ].join(",");
    fs.writeFileSync(OUT_FILE, header + "\n", "utf8");
  }
}

function appendRows(rows) {
  if (!rows.length) return;
  const lines = rows
    .map((fields) => fields.map(csvEscape).join(","))
    .join("\n");
  fs.appendFileSync(OUT_FILE, lines + "\n", "utf8");
}

loadSeen();
writeHeaderIfNeeded();

async function fetchAndSave() {
  try {
    const feed = await parser.parseURL(FEED_URL);
    if (!feed.items || feed.items.length === 0) {
      console.log("[INFO] Pas de news dans le flux.");
      return;
    }

    const newRows = [];
    for (const item of feed.items) {
      const dtRaw = new Date(item.isoDate || item.pubDate || Date.now());
      const parisTime = parisIso(dtRaw);
      const utcIso = dtRaw.toISOString();
      const title = (item.title || "").trim();
      const link = (item.link || "").trim();
      const summary = (item.contentSnippet || "").trim();
      const source = (item.creator || item.author || item.source || "").trim();
      const signature = `${utcIso}|${title}`;

      if (seenSet.has(signature)) continue;
      remember(signature);

      newRows.push([
        parisTime,
        utcIso,
        title,
        link,
        summary,
        source,
        signature,
      ]);
      console.log(`[NEWS] ${parisTime} > ${title}`);
    }

    if (newRows.length > 0) {
      appendRows(newRows.reverse());
      persistSeen();
      console.log(`[SAVED] ${newRows.length} nouvelles news ajoutées.`);
    }
  } catch (err) {
    console.error("[ERROR] RSS fetch failed:", err.message);
  }
}

await fetchAndSave();
setInterval(fetchAndSave, INTERVAL_MS);
