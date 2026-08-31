import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { mkdir, writeFile, stat, rm } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { createClient } from 'redis';

const ROOT = path.resolve(process.cwd(), '..');
const SESSION_ROOT = process.env.DAGFLOW_SESSION_ROOT || '/tmp/dagflow-sessions';
const PORT = Number(process.env.PORT || 3000);
const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379';
const MAX_ACTIVE_JOBS = Number(process.env.DAGFLOW_MAX_ACTIVE_JOBS || 1);
const SESSION_TTL_MS = Number(process.env.DAGFLOW_SESSION_TTL_MS || 24 * 60 * 60 * 1000);
const SESSION_LIMIT_WINDOW_MS = Number(process.env.DAGFLOW_SESSION_LIMIT_WINDOW_MS || 60 * 60 * 1000);
const SESSION_LIMIT_MAX = Number(process.env.DAGFLOW_SESSION_LIMIT_MAX || 10);

const PROVIDERS = {
  'ssec-litellm': {
    models: new Set(['gemma-4-31b', 'gpt-5.4-mini', 'kimi-k2-thinking'])
  },
  openai: {
    models: new Set(['gpt-4o', 'gpt-5.4-mini', 'gpt-5.4-nano'])
  },
  anthropic: {
    models: new Set(['claude-sonnet-4-20250514'])
  }
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

  if (!provider || !PROVIDERS[provider]) {
    throw new Error('invalid provider');
  }
  if (!model || !PROVIDERS[provider].models.has(model)) {
    throw new Error('invalid model for provider');
  }
  if (!apiKey) {
    throw new Error('api key is required');
  }
  if (!Number.isInteger(rows) || rows <= 0 || rows > 100000) {
    throw new Error('rows must be an integer between 1 and 100000');
  }

  return { provider, model, apiKey, description, objectives, rows };
}

async function enforceRateLimit() {
  const bucket = Math.floor(Date.now() / SESSION_LIMIT_WINDOW_MS);
  const key = `dagflow:session-limit:${bucket}`;
  const count = await redis.incr(key);
  if (count === 1) {
    await redis.expire(key, Math.ceil(SESSION_LIMIT_WINDOW_MS / 1000));
  }
  if (count > SESSION_LIMIT_MAX) {
    throw new Error('rate limit exceeded');
  }
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
  text(res, 404, 'not found');
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

  setTimeout(() => {
    cleanupSession(id).catch(() => {});
  }, SESSION_TTL_MS).unref?.();

  return manifest;
}

async function getSession(id) {
  const data = await redis.hGetAll(`session:${id}`);
  return Object.keys(data).length ? data : null;
}

async function buildArchive(session) {
  const archiveContent = {
    sessionId: session.id,
    provider: session.provider,
    model: session.model,
    logFile: 'logs.txt',
    manifestFile: 'manifest.json'
  };

  await writeFile(path.join(session.workspace, 'archive-manifest.json'), JSON.stringify(archiveContent, null, 2));

  const zipArgs = [
    '-r',
    session.archive,
    'synthdata',
    'manifest.json',
    'archive-manifest.json',
    'logs.txt'
  ];

  await new Promise((resolve, reject) => {
    const child = spawn('zip', zipArgs, { cwd: session.workspace });
    child.on('error', reject);
    child.on('close', (code) => code === 0 ? resolve() : reject(new Error(`zip failed with code ${code}`)));
  });
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

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host}`);

  if (req.method === 'GET' && url.pathname === '/') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DagFlow</title>
    <style>
      body { font-family: Inter, system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
      main { max-width: 920px; margin: 0 auto; padding: 32px 20px 48px; }
      .card { background: #111827; border: 1px solid #243041; border-radius: 18px; padding: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.25); }
      h1 { margin: 0 0 8px; font-size: 32px; }
      p.subtle { margin-top: 0; color: #94a3b8; }
      form { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      label { display: grid; gap: 6px; font-size: 14px; color: #cbd5e1; }
      input, textarea, select, button { border-radius: 12px; border: 1px solid #334155; background: #0b1220; color: #e2e8f0; padding: 12px 14px; font: inherit; }
      textarea { min-height: 92px; resize: vertical; }
      .full { grid-column: 1 / -1; }
      button { background: linear-gradient(135deg, #2563eb, #7c3aed); border: 0; cursor: pointer; font-weight: 600; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .panel { margin-top: 18px; padding: 16px; border: 1px solid #243041; border-radius: 14px; background: #0b1220; }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
      a { color: #93c5fd; }
      @media (max-width: 720px) { form { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <main>
      <h1>DagFlow</h1>
      <p class="subtle">Create a synthetic dataset package with your own provider API key.</p>
      <div class="card">
        <form id="f">
          <label>
            Provider
            <select name="provider" id="provider">
              <option value="openai">OpenAI</option>
              <option value="ssec-litellm">SSEC LiteLLM</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </label>
          <label>
            Model
            <select name="model" id="model"></select>
          </label>
          <label class="full">
            API Key
            <input name="apiKey" type="password" autocomplete="off" />
          </label>
          <label class="full">
            Description
            <textarea name="description" placeholder="What should the dataset represent?"></textarea>
          </label>
          <label class="full">
            Objectives
            <textarea name="objectives" placeholder="What should users be able to analyze?"></textarea>
          </label>
          <label>
            Rows
            <input name="rows" type="number" min="1" max="100000" value="1000" />
          </label>
          <div style="display:flex; align-items:end;">
            <button type="submit" id="submit">Start</button>
          </div>
        </form>
      </div>
      <div class="panel">
        <strong id="meta">Ready.</strong>
        <p><a id="download" hidden>Download package</a></p>
        <pre id="log"></pre>
      </div>
    </main>
    <script>
      const modelMap = {
        'openai': ['gpt-4o', 'gpt-5.4-mini', 'gpt-5.4-nano'],
        'ssec-litellm': ['gemma-4-31b', 'gpt-5.4-mini', 'kimi-k2-thinking'],
        'anthropic': ['claude-sonnet-4-20250514']
      };
      const f = document.getElementById('f');
      const provider = document.getElementById('provider');
      const model = document.getElementById('model');
      const submit = document.getElementById('submit');
      const meta = document.getElementById('meta');
      const log = document.getElementById('log');
      const download = document.getElementById('download');
      let sessionId = null;
      let timer = null;

      function syncModels() {
        const options = modelMap[provider.value] || [];
        model.innerHTML = options.map((v) => '<option value="' + v + '">' + v + '</option>').join('');
      }

      provider.addEventListener('change', syncModels);
      syncModels();

      async function refresh() {
        if (!sessionId) return;
        const statusRes = await fetch('/sessions/' + sessionId);
        const status = await statusRes.json();
        meta.textContent = 'Session ' + status.id + ': ' + status.status;
        const logRes = await fetch('/sessions/' + sessionId + '/logs');
        log.textContent = await logRes.text();
        if (status.status === 'completed') {
          download.hidden = false;
          download.href = '/sessions/' + sessionId + '/download';
          download.textContent = 'Download package';
          clearInterval(timer);
        }
        if (status.status === 'error') {
          if (status.error) {
            log.textContent = log.textContent + '\n\nError: ' + status.error;
          }
          clearInterval(timer);
        }
        if (status.status === 'expired') {
          download.hidden = true;
          clearInterval(timer);
        }
      }

      f.onsubmit = async (e) => {
        e.preventDefault();
        submit.disabled = true;
        const body = Object.fromEntries(new FormData(f).entries());
        body.rows = Number(body.rows || 1000);
        try {
          const res = await fetch('/sessions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
          const payload = await res.json();
          if (!res.ok) {
            throw new Error(payload.error || 'failed to start session');
          }
          sessionId = payload.id;
          meta.textContent = `Session ${payload.id}: queued`;
          download.hidden = true;
          log.textContent = '';
          clearInterval(timer);
          timer = setInterval(refresh, 2000);
          refresh();
        } catch (err) {
          meta.textContent = err.message;
        } finally {
          submit.disabled = false;
        }
      };
    </script>
  </body>
</html>`);
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

await ensureDir(SESSION_ROOT);
server.listen(PORT, () => {
  console.log(`DagFlow web listening on http://localhost:${PORT}`);
});
