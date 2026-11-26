import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ResearchHistory } from '../../src/components/ResearchHistory';
import { apiService } from '../../src/services/api';
import type { QueryHistory } from '../../src/types';

vi.mock('../../src/services/api', () => ({
  apiService: { getQueryHistory: vi.fn(), getRun: vi.fn() },
}));

const getQueryHistory = vi.mocked(apiService.getQueryHistory);

const item = (id: number): QueryHistory => ({
  id,
  run_id: `run-${id}`,
  thread_id: `thread-${id}-aaaaaaaaaaaa`,
  query: `question ${id}`,
  status: 'completed',
  intent: 'RESEARCH',
  execution_time: 12.3,
  created_at: '2026-11-14T10:00:00Z',
});

beforeEach(() => {
  getQueryHistory.mockResolvedValue([item(1), item(2)]);
});

describe('ResearchHistory', () => {
  it('lists the runs returned by the backend', async () => {
    render(<ResearchHistory />);
    expect(await screen.findByText('question 1')).toBeInTheDocument();
    expect(screen.getByText('question 2')).toBeInTheDocument();
    expect(screen.getByText('Showing 2 of 2 queries')).toBeInTheDocument();
  });

  it('only offers status filters the backend can produce', async () => {
    render(<ResearchHistory />);
    await screen.findByText('question 1');
    const options = screen.getAllByRole('option').map((o) => o.textContent);
    expect(options).not.toContain('Processing');
    expect(options).toEqual(expect.arrayContaining(['All statuses', 'Completed', 'Failed']));
    expect(screen.getAllByText('Research')).toHaveLength(2);
    expect(screen.queryByText('RESEARCH')).not.toBeInTheDocument();
  });

  it('has no Load More button that does nothing', async () => {
    getQueryHistory.mockResolvedValue(Array.from({ length: 25 }, (_, i) => item(i + 1)));
    render(<ResearchHistory />);
    await screen.findByText('question 1');
    expect(screen.queryByRole('button', { name: 'Load More' })).not.toBeInTheDocument();
  });

  it('shows a failure instead of an empty history and retries on refresh', async () => {
    getQueryHistory.mockRejectedValueOnce(new Error('offline'));
    render(<ResearchHistory />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load research history');
    expect(screen.queryByText(/No research history yet/)).not.toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByText('question 1')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
