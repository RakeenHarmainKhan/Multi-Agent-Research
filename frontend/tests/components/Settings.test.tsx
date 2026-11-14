import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Settings } from '../../src/components/Settings';
import { apiService } from '../../src/services/api';

vi.mock('../../src/services/api', () => ({
  apiService: { checkHealth: vi.fn() },
}));

const checkHealth = vi.mocked(apiService.checkHealth);

beforeEach(() => {
  checkHealth.mockResolvedValue({
    status: 'healthy',
    version: '1.0.0',
    uptime: 5,
    database: 'in-memory',
    llm_service: 'configured',
    timestamp: '2026-11-14T00:00:00Z',
  });
});

describe('Settings', () => {
  it('shows the real backend configuration', async () => {
    render(<Settings />);
    expect(await screen.findByText('Online · v1.0.0')).toBeInTheDocument();
    expect(screen.getByText('Configured')).toBeInTheDocument();
    expect(screen.getByText('in-memory')).toBeInTheDocument();
  });

  it('has no API key, model or inference inputs that the backend would ignore', async () => {
    const { container } = render(<Settings />);
    await screen.findByText('Online · v1.0.0');
    expect(container.querySelector('input')).toBeNull();
    expect(screen.queryByText(/OpenRouter/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /save/i })).not.toBeInTheDocument();
  });

  it('reports an unreachable backend', async () => {
    checkHealth.mockRejectedValue(new Error('down'));
    render(<Settings />);
    expect(await screen.findByText('Unreachable')).toBeInTheDocument();
  });
});
