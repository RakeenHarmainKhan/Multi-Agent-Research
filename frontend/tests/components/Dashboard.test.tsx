import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Dashboard } from '../../src/components/Dashboard';
import { apiService } from '../../src/services/api';
import type { HealthStatus, SystemStats } from '../../src/types';

vi.mock('../../src/services/api', () => ({
  apiService: { getSystemStats: vi.fn(), checkHealth: vi.fn() },
}));

const getSystemStats = vi.mocked(apiService.getSystemStats);
const checkHealth = vi.mocked(apiService.checkHealth);

const STATS: SystemStats = {
  total_queries: 4,
  successful_queries: 3,
  failed_queries: 1,
  avg_latency: 21.5,
  total_tokens: 12345,
  avg_tokens_per_query: 3086.3,
  total_cost: null,
  queries_today: 4,
};

const HEALTH: HealthStatus = {
  status: 'healthy',
  version: '1.0.0',
  uptime: 12,
  database: 'in-memory',
  llm_service: 'configured',
  timestamp: '2026-11-14T00:00:00Z',
};

beforeEach(() => {
  getSystemStats.mockResolvedValue(STATS);
  checkHealth.mockResolvedValue(HEALTH);
});

describe('Dashboard', () => {
  it('shows real token usage', async () => {
    render(<Dashboard />);
    expect(await screen.findByText('12,345')).toBeInTheDocument();
    expect(screen.getByText('3,086 per query')).toBeInTheDocument();
  });

  it('shows a dash, not $0.00, when pricing is not configured', async () => {
    render(<Dashboard />);
    await screen.findByText('Est. Cost');
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('$0.00')).not.toBeInTheDocument();
  });

  it('shows the cost when the backend reports one', async () => {
    getSystemStats.mockResolvedValue({ ...STATS, total_cost: 1.5 });
    render(<Dashboard />);
    expect(await screen.findByText('$1.50')).toBeInTheDocument();
  });

  it('reads LLM and storage status from /health instead of hardcoding it', async () => {
    checkHealth.mockResolvedValue({ ...HEALTH, llm_service: 'not_configured' });
    render(<Dashboard />);
    expect(await screen.findByText('Not configured')).toBeInTheDocument();
    expect(screen.getByText('in-memory (resets on restart)')).toBeInTheDocument();
  });

  it('marks the API as unreachable when /health fails', async () => {
    checkHealth.mockRejectedValue(new Error('down'));
    render(<Dashboard />);
    expect(await screen.findByText('Unreachable')).toBeInTheDocument();
  });

  it('does not show the removed fake metrics', async () => {
    render(<Dashboard />);
    await screen.findByText('Tokens Used');
    expect(screen.queryByText(/active users/i)).not.toBeInTheDocument();
    expect(screen.queryByText('Database')).not.toBeInTheDocument();
  });
});
