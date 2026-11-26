import { beforeEach, describe, expect, it, vi } from 'vitest';

const client = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  interceptors: { response: { use: vi.fn() } },
}));
const created = vi.hoisted(() => ({ config: undefined as Record<string, unknown> | undefined }));
const create = vi.hoisted(() =>
  vi.fn((config: Record<string, unknown>) => {
    created.config = config;
    return client;
  })
);

vi.mock('axios', () => ({ default: { create } }));

import { apiService } from '../../src/services/api';

beforeEach(() => {
  client.get.mockReset();
  client.post.mockReset();
});

describe('ApiService', () => {
  it('targets the /api/v1 base URL with a long timeout', () => {
    expect(created.config?.baseURL).toMatch(/\/api\/v1$/);
    expect(created.config?.timeout).toBe(300000);
  });

  it('executeResearch streams stages and resolves with the result event', async () => {
    const body =
      'event: stage\ndata: {"stage": "input_guard"}\n\n' +
      'event: stage\ndata: {"stage": "writer"}\n\n' +
      'event: result\ndata: {"thread_id": "t", "report": "r"}\n\n';
    const fetchMock = vi.fn().mockResolvedValue(new Response(body));
    vi.stubGlobal('fetch', fetchMock);
    const stages: string[] = [];

    const out = await apiService.executeResearch({ query: 'q', mode: 'research' }, (s) => stages.push(s));

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/v1\/research\/stream$/);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ query: 'q', mode: 'research' });
    expect(stages).toEqual(['input_guard', 'writer']);
    expect(out).toEqual({ thread_id: 't', report: 'r' });
  });

  it('executeResearch rejects with the server detail on an error event', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('event: error\ndata: {"detail": "boom"}\n\n')));
    await expect(apiService.executeResearch({ query: 'q' })).rejects.toThrow('boom');
  });

  it('executeResearch rejects on a non-ok status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 422 })));
    await expect(apiService.executeResearch({ query: 'q' })).rejects.toThrow('422');
  });

  it('getResearchResult fetches by thread id', async () => {
    client.get.mockResolvedValue({ data: { thread_id: 'abc' } });
    await apiService.getResearchResult('abc');
    expect(client.get).toHaveBeenCalledWith('/research/abc');
  });

  it('getRun fetches one run by id', async () => {
    client.get.mockResolvedValue({ data: { thread_id: 'abc' } });
    await apiService.getRun('run-1');
    expect(client.get).toHaveBeenCalledWith('/runs/run-1');
  });

  it('getSystemStats and getQueryHistory use the admin routes', async () => {
    client.get.mockResolvedValue({ data: [] });
    await apiService.getSystemStats();
    await apiService.getQueryHistory(5);
    expect(client.get).toHaveBeenNthCalledWith(1, '/admin/stats');
    expect(client.get).toHaveBeenNthCalledWith(2, '/admin/history', { params: { limit: 5 } });
  });

  it('checkHealth calls /health', async () => {
    client.get.mockResolvedValue({ data: { status: 'healthy' } });
    expect(await apiService.checkHealth()).toEqual({ status: 'healthy' });
    expect(client.get).toHaveBeenCalledWith('/health');
  });
});
