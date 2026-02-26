const DEFAULT_PROXY_TARGET = "https://visionsort-ai.onrender.com";

const EXACT_ALLOWED_PATHS = new Set([
  "/",
  "/api",
  "/upload",
  "/api/upload",
  "/upload/async",
  "/api/upload/async",
  "/download/zip",
  "/api/download/zip",
]);

const PREFIX_ALLOWED_PATHS = ["/jobs/", "/api/jobs/"];

function normalizeProxyTarget(rawTarget) {
  const value = String(rawTarget || "").trim() || DEFAULT_PROXY_TARGET;
  try {
    const parsed = new URL(value);
    if (!["https:", "http:"].includes(parsed.protocol)) {
      return "";
    }
    return `${parsed.origin}${parsed.pathname}`.replace(/\/+$/, "");
  } catch {
    return "";
  }
}

function extractPathFromRequest(req) {
  const raw = req.query?.path;
  const parts = Array.isArray(raw) ? raw : raw ? [raw] : [];
  const normalized = parts
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .map((part) => part.replace(/^\/+|\/+$/g, ""));

  if (!normalized.length) {
    return "/";
  }

  return `/${normalized.join("/")}`.replace(/\/{2,}/g, "/");
}

function isAllowedPath(pathname) {
  if (EXACT_ALLOWED_PATHS.has(pathname)) {
    return true;
  }
  return PREFIX_ALLOWED_PATHS.some((prefix) => pathname.startsWith(prefix));
}

async function readRawBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return chunks.length ? Buffer.concat(chunks) : null;
}

function copyUpstreamHeaders(upstreamResponse, res) {
  const headerNames = [
    "content-type",
    "content-disposition",
    "cache-control",
    "x-request-id",
    "retry-after",
  ];

  headerNames.forEach((name) => {
    const value = upstreamResponse.headers.get(name);
    if (value) {
      res.setHeader(name, value);
    }
  });
}

module.exports = async function handler(req, res) {
  const method = String(req.method || "GET").toUpperCase();
  if (method === "OPTIONS") {
    return res.status(204).end();
  }

  const targetBase = normalizeProxyTarget(process.env.BACKEND_PROXY_TARGET || DEFAULT_PROXY_TARGET);
  if (!targetBase) {
    return res.status(500).json({
      error: {
        code: "PROXY_CONFIG_ERROR",
        message: "Invalid backend proxy target configuration.",
      },
    });
  }

  const upstreamPath = extractPathFromRequest(req);
  if (!isAllowedPath(upstreamPath)) {
    return res.status(404).json({
      error: {
        code: "NOT_FOUND",
        message: "Route not available through proxy.",
      },
    });
  }

  const queryIndex = String(req.url || "").indexOf("?");
  const queryString = queryIndex >= 0 ? String(req.url).slice(queryIndex) : "";
  const upstreamUrl = `${targetBase}${upstreamPath}${queryString}`;

  const headers = {};
  const contentType = req.headers["content-type"];
  const accept = req.headers.accept;
  const requestId = req.headers["x-request-id"];

  if (contentType) {
    headers["content-type"] = contentType;
  }
  if (accept) {
    headers.accept = accept;
  }
  if (requestId) {
    headers["x-request-id"] = requestId;
  }

  const apiKey = String(process.env.BACKEND_API_KEY || "").trim();
  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  const requestOptions = {
    method,
    headers,
    redirect: "manual",
  };

  if (!["GET", "HEAD"].includes(method)) {
    requestOptions.body = await readRawBody(req);
  }

  try {
    const upstreamResponse = await fetch(upstreamUrl, requestOptions);
    const body = Buffer.from(await upstreamResponse.arrayBuffer());

    copyUpstreamHeaders(upstreamResponse, res);
    return res.status(upstreamResponse.status).send(body);
  } catch {
    return res.status(502).json({
      error: {
        code: "UPSTREAM_UNAVAILABLE",
        message: "Backend service is unavailable.",
      },
    });
  }
};
