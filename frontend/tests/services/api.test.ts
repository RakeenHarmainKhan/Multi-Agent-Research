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

  it('executeResearch posts the request and returns the data', async () => {
    client.post.mockResolvedValue({ data: { thread_id: 't', report: 'r' } });
    const out = await apiService.executeResearch({ query: 'q', mode: 'research' });
    expect(client.post).toHaveBeenCalledWith('/research', { query: 'q', mode: 'research' });
    expect(out).toEqual({ thread_id: 't', report: 'r' });
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

  it('submitFeedback posts to the feedback route', async () => {
    client.post.mockResolvedValue({ data: {} });
    await apiService.submitFeedback('abc', { feedback: 'x', approved: true });
    expect(client.post).toHaveBeenCalledWith('/research/abc/feedback', { feedback: 'x', approved: true });
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
