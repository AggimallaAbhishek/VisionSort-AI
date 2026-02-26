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

function parsePositiveInt(rawValue, fallback) {
  const parsed = Number.parseInt(String(rawValue || "").trim(), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

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

function buildUpstreamQueryString(req) {
  const params = new URLSearchParams();
  const query = req.query || {};

  for (const [key, value] of Object.entries(query)) {
    if (key === "path") {
      continue;
    }

    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== undefined && item !== null) {
          params.append(key, String(item));
        }
      });
      continue;
    }

    if (value !== undefined && value !== null) {
      params.set(key, String(value));
    }
  }

  const qs = params.toString();
  return qs ? `?${qs}` : "";
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
  const upstreamTimeoutMs = parsePositiveInt(process.env.PROXY_UPSTREAM_TIMEOUT_MS, 9000);

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

  const queryString = buildUpstreamQueryString(req);
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

  const controller = new AbortController();
  const timeoutHandle = setTimeout(() => controller.abort(), upstreamTimeoutMs);

  const requestOptions = {
    method,
    headers,
    redirect: "manual",
    signal: controller.signal,
  };

  if (!["GET", "HEAD"].includes(method)) {
    // Stream multipart/file uploads through the proxy to reduce memory usage.
    requestOptions.body = req;
    requestOptions.duplex = "half";
  }

  try {
    const upstreamResponse = await fetch(upstreamUrl, requestOptions);
    const body = Buffer.from(await upstreamResponse.arrayBuffer());

    copyUpstreamHeaders(upstreamResponse, res);
    res.setHeader("x-proxy-timeout-ms", String(upstreamTimeoutMs));
    return res.status(upstreamResponse.status).send(body);
  } catch (error) {
    const isTimeout = error instanceof DOMException && error.name === "AbortError";
    return res.status(isTimeout ? 504 : 502).json({
      error: {
        code: isTimeout ? "UPSTREAM_TIMEOUT" : "UPSTREAM_UNAVAILABLE",
        message: isTimeout
          ? "Backend did not respond before proxy timeout."
          : "Backend service is unavailable.",
      },
    });
  } finally {
    clearTimeout(timeoutHandle);
  }
};
