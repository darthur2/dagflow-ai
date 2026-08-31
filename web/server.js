import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { mkdir, writeFile, stat, rm, readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { createClient } from 'redis';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const PORT = Number(process.env.PORT || 3000);
const SESSION_ROOT = process.env.DAGFLOW_SESSION_ROOT || '/tmp/dagflow-sessions';
const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379';
const SESSION_TTL_MS = Number(process.env.DAGFLOW_SESSION_TTL_MS || 24 * 60 * 60 * 1000);
const SESSION_LIMIT_WINDOW_MS = Number(process.env.DAGFLOW_SESSION_LIMIT_WINDOW_MS || 60 * 60 * 1000);
const SESSION_LIMIT_MAX = Number(process.env.DAGFLOW_SESSION_LIMIT_MAX || 10);

const PROVIDERS = {
  'ssec-litellm': { models: new Set(['gemma-4-31b', 'gpt-5.4-mini', 'kimi-k2-thinking']) },
  openai: { models: new Set(['gpt-4o', 'gpt-5.4-mini', 'gpt-5.4-nano']) },
  anthropic: { models: new Set(['claude-sonnet-4-20250514']) }
};

const redis = createClient({ url: REDIS_URL });
await redis.connect();

async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
}

async function readJson(req) {
  let body = '';
  for await (const chunk of req) body += chunk;
  return body ? JSON.parse(body) : {};
}

function validateSessionInput(body) {
  const provider = String(body.provider || '').trim();
  const model = String(body.model || '').trim();
  const apiKey = String(body.apiKey || '').trim();
  const description = String(body.description || '').trim();
  const objectives = String(body.objectives || '').trim();
  const rows = Number(body.rows || 1000);

  if (!provider || !PROVIDERS[provider]) throw new Error('invalid provider');
  if (!model || !PROVIDERS[provider].models.has(model)) throw new Error('invalid model for provider');
  if (!apiKey) throw new Error('api key is required');
  if (!Number.isInteger(rows) || rows <= 0 || rows > 100000) throw new Error('rows must be an integer between 1 and 100000');

  return { provider, model, apiKey, description, objectives, rows };
}

async function enforceRateLimit() {
  const bucket = Math.floor(Date.now() / SESSION_LIMIT_WINDOW_MS);
  const key = `dagflow:session-limit:${bucket}`;
  const count = await redis.incr(key);
  if (count === 1) await redis.expire(key, Math.ceil(SESSION_LIMIT_WINDOW_MS / 1000));
  if (count > SESSION_LIMIT_MAX) throw new Error('rate limit exceeded');
}

function json(res, status, payload) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(payload));
}

function text(res, status, body, headers = {}) {
  res.writeHead(status, { 'content-type': 'text/plain; charset=utf-8', ...headers });
  res.end(body);
}

function notFound(res) {
  res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
  res.end('not found');
}

async function serveIndex(res) {
  const html = await readFile(path.join(ROOT, 'web', 'index.html'), 'utf8');
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
}

function sessionPaths(id) {
  const base = path.join(SESSION_ROOT, id);
  return {
    base,
    synthdata: path.join(base, 'synthdata'),
    logs: path.join(base, 'logs.txt'),
    manifest: path.join(base, 'manifest.json'),
    archive: path.join(base, 'synthetic-dataset-package.zip')
  };
}

async function createSession({ provider, model, apiKey, description, objectives, rows }) {
  await enforceRateLimit();
  const id = randomUUID();
  const p = sessionPaths(id);
  await ensureDir(p.synthdata);

  const manifest = {
    id,
    provider,
    model,
    description: description || '',
    objectives: objectives || '',
    rows: rows || 1000,
    createdAt: new Date().toISOString(),
    status: 'queued'
  };

  await writeFile(p.manifest, JSON.stringify(manifest, null, 2));

  await redis.hSet(`session:${id}`, {
    id,
    provider,
    model,
    description: manifest.description,
    objectives: manifest.objectives,
    rows: String(manifest.rows),
    createdAt: manifest.createdAt,
    status: 'queued',
    apiKey,
    workspace: p.base,
    manifest: p.manifest,
    logs: p.logs,
    archive: p.archive,
    expiresAt: String(Date.now() + SESSION_TTL_MS)
  });

  await redis.rPush('dagflow:queue', id);
  setTimeout(() => { cleanupSession(id).catch(() => {}); }, SESSION_TTL_MS).unref?.();
  return { id };
}

async function getSession(id) {
  const data = await redis.hGetAll(`session:${id}`);
  return Object.keys(data).length ? data : null;
}

async function cleanupSession(id) {
  const session = await getSession(id);
  if (!session) return;
  await redis.hSet(`session:${id}`, { status: 'expired' });
  await redis.del(`session:${id}`, `session:${id}:log`);
  await rm(session.workspace, { recursive: true, force: true });
}

async function sendFile(res, filePath, downloadName) {
  const fh = await stat(filePath);
  res.writeHead(200, {
    'content-type': 'application/zip',
    'content-length': fh.size,
    'content-disposition': `attachment; filename="${downloadName}"`
  });
  await pipeline(createReadStream(filePath), res);
}

async function main() {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);

    if (req.method === 'GET' && url.pathname === '/') {
      await serveIndex(res);
      return;
    }

    if (req.method === 'POST' && url.pathname === '/sessions') {
      try {
        const body = validateSessionInput(await readJson(req));
        const created = await createSession(body);
        json(res, 201, created);
      } catch (err) {
        json(res, 400, { error: err.message });
      }
      return;
    }

    const sessionMatch = url.pathname.match(/^\/sessions\/([^/]+)$/);
    if (req.method === 'GET' && sessionMatch) {
      const session = await getSession(sessionMatch[1]);
      if (!session) return notFound(res);
      json(res, 200, {
        id: session.id,
        provider: session.provider,
        model: session.model,
        status: session.status,
        createdAt: session.createdAt,
        error: session.error || null,
        expiresAt: session.expiresAt || null
      });
      return;
    }

    const logMatch = url.pathname.match(/^\/sessions\/([^/]+)\/logs$/);
    if (req.method === 'GET' && logMatch) {
      const session = await getSession(logMatch[1]);
      if (!session) return notFound(res);
      text(res, 200, (await redis.get(`session:${session.id}:log`)) || '');
      return;
    }

    const downloadMatch = url.pathname.match(/^\/sessions\/([^/]+)\/download$/);
    if (req.method === 'GET' && downloadMatch) {
      const session = await getSession(downloadMatch[1]);
      if (!session) return notFound(res);
      if (session.status !== 'completed') return text(res, 409, 'session not completed');
      return sendFile(res, session.archive, `dagflow-session-${session.id}.zip`);
    }

    return notFound(res);
  });

  server.listen(PORT, () => {
    console.log(`DagFlow web listening on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
