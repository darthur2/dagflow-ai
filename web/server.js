import http from 'node:http';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..');
const PORT = Number(process.env.PORT || 3000);

function notFound(res) {
  res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
  res.end('not found');
}

async function serveIndex(res) {
  const html = await import('node:fs/promises').then(({ readFile }) => readFile(path.join(ROOT, 'web', 'index.html'), 'utf8'));
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
}

async function main() {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);

    if (req.method === 'GET' && url.pathname === '/') {
      await serveIndex(res);
      return;
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
