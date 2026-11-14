import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from '../src/App';

vi.mock('../src/services/api', () => ({
  apiService: {
    executeResearch: vi.fn(),
    submitFeedback: vi.fn(),
    getSystemStats: vi.fn().mockReturnValue(new Promise(() => {})),
    checkHealth: vi.fn().mockReturnValue(new Promise(() => {})),
    getQueryHistory: vi.fn().mockResolvedValue([]),
  },
}));

describe('App tab persistence', () => {
  it('opens the chat tab by default', () => {
    render(<App />);
    expect(screen.getByText(/Welcome to MARA/)).toBeVisible();
  });

  it('restores the saved tab on first render', () => {
    localStorage.setItem('mara_active_tab', 'settings');
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Settings' })).toBeInTheDocument();
  });

  it('ignores an invalid saved tab', () => {
    localStorage.setItem('mara_active_tab', 'nonsense');
    render(<App />);
    expect(screen.getByText(/Welcome to MARA/)).toBeVisible();
  });

  it('saves the tab when it changes', async () => {
    render(<App />);
    await userEvent.setup().click(screen.getByRole('button', { name: 'Settings' }));
    expect(localStorage.getItem('mara_active_tab')).toBe('settings');
  });
});
