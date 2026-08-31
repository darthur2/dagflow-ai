import { mkdir, writeFile, rm } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { createClient } from 'redis';

const ROOT = path.resolve(process.cwd(), '..');
const SESSION_ROOT = process.env.DAGFLOW_SESSION_ROOT || '/tmp/dagflow-sessions';
const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379';
const MAX_ACTIVE_JOBS = Number(process.env.DAGFLOW_MAX_ACTIVE_JOBS || 1);
const SESSION_TTL_MS = Number(process.env.DAGFLOW_SESSION_TTL_MS || 24 * 60 * 60 * 1000);

const redis = createClient({ url: REDIS_URL });
const activeJobs = new Set();
const HEALTH_PORT = Number(process.env.WORKER_HEALTH_PORT || 3999);

await redis.connect();

async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
}

async function getSession(id) {
  const data = await redis.hGetAll(`session:${id}`);
  return Object.keys(data).length ? data : null;
}

async function appendLog(id, chunk) {
  const key = `session:${id}:log`;
  const current = await redis.get(key) || '';
  await redis.set(key, current + chunk);
}

async function setStatus(id, status, extra = {}) {
  await redis.hSet(`session:${id}`, { status, ...extra });
}

async function buildArchive(session) {
  await writeFile(path.join(session.workspace, 'archive-manifest.json'), JSON.stringify({
    sessionId: session.id,
    provider: session.provider,
    model: session.model,
    logFile: 'logs.txt',
    manifestFile: 'manifest.json'
  }, null, 2));

  const { execFile } = await import('node:child_process');
  await new Promise((resolve, reject) => {
    const child = execFile('zip', ['-r', session.archive, 'synthdata', 'manifest.json', 'archive-manifest.json', 'logs.txt'], { cwd: session.workspace }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`zip failed: ${stderr || error.message}`));
        return;
      }
      resolve();
    });
    child.stdout?.on('data', (buf) => appendLog(session.id, buf.toString()));
    child.stderr?.on('data', (buf) => appendLog(session.id, buf.toString()));
  });
}

async function runSession(id) {
  const session = await getSession(id);
  if (!session) return;

  await setStatus(id, 'running');
  await writeFile(session.logs, '');

  const { execFile } = await import('node:child_process');
  await new Promise((resolve, reject) => {
    const child = execFile('R', ['-e', `source('R/utils/generate_data.R'); generate_data(n=${Number(session.rows || 1000)}, output_path='synthdata/generated_data.csv')`], { cwd: ROOT, env: process.env }, (error) => {
      if (error) {
        reject(new Error(`generation failed: ${error.message}`));
        return;
      }
      resolve();
    });
    child.stdout?.on('data', (buf) => appendLog(id, buf.toString()));
    child.stderr?.on('data', (buf) => appendLog(id, buf.toString()));
  });

  await setStatus(id, 'completed');
  const log = await redis.get(`session:${id}:log`) || '';
  await writeFile(session.logs, log);
  await buildArchive({ ...session, log });
}

async function cleanupSession(id) {
  const session = await getSession(id);
  if (!session) return;
  await setStatus(id, 'expired');
  await redis.del(`session:${id}:log`);
  await rm(session.workspace, { recursive: true, force: true });
}

async function drainQueue() {
  while (activeJobs.size < MAX_ACTIVE_JOBS) {
    const id = await redis.lPop('dagflow:queue');
    if (!id) break;
    const session = await getSession(id);
    if (!session || session.status !== 'queued') continue;
    activeJobs.add(id);
    runSession(id)
      .catch(async (err) => {
        await redis.hSet(`session:${id}`, { status: 'error' });
        await appendLog(id, `\n[web] ${err.stack || err.message}\n`);
      })
      .finally(() => {
        activeJobs.delete(id);
        drainQueue().catch(() => {});
      });
  }
}

async function main() {
  await ensureDir(SESSION_ROOT);

  const healthServer = http.createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
      try {
        await redis.ping();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, redis: 'up', activeJobs: activeJobs.size }));
      } catch (err) {
        res.writeHead(503, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: err.message }));
      }
      return;
    }

    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end('not found');
  });

  healthServer.listen(HEALTH_PORT, '0.0.0.0');

  setInterval(() => {
    drainQueue().catch(() => {});
  }, 3000).unref();

  setInterval(async () => {
    const keys = await redis.keys('session:*');
    for (const key of keys) {
      if (key.endsWith(':log')) continue;
      const session = await redis.hGetAll(key);
      if (!session.id) continue;
      const age = Date.now() - Date.parse(session.createdAt || new Date().toISOString());
      if (age > SESSION_TTL_MS) {
        await cleanupSession(session.id);
      }
    }
  }, 60 * 60 * 1000).unref();

  setInterval(async () => {
    const sessions = await redis.keys('session:*');
    const live = new Set(await redis.lRange('dagflow:queue', 0, -1));
    for (const key of sessions) {
      if (key.endsWith(':log')) continue;
      const session = await redis.hGetAll(key);
      if (!session.id) continue;
      if (session.status === 'running' && !activeJobs.has(session.id) && !live.has(session.id)) {
        await setStatus(session.id, 'error', { error: 'worker restarted before completion' });
      }
    }
  }, 15000).unref();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
