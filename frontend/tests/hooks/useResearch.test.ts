import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useResearch } from '../../src/hooks/useResearch';
import { apiService } from '../../src/services/api';

vi.mock('../../src/services/api', () => ({
  apiService: { executeResearch: vi.fn(), submitFeedback: vi.fn() },
}));

const executeResearch = vi.mocked(apiService.executeResearch);
const submitFeedback = vi.mocked(apiService.submitFeedback);

const RESPONSE = {
  thread_id: 'thread-1',
  status: 'completed' as const,
  report: 'Report [1]',
  citations: [{ id: 1, type: 'web' as const, title: 'T', url: 'http://t', snippet: 's' }],
  sources: [],
};

beforeEach(() => {
  executeResearch.mockResolvedValue(RESPONSE);
});

describe('executeResearch', () => {
  it('adds the user and assistant messages and stores citations', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('lora'));
    expect(result.current.messages.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(result.current.messages[1].content).toBe('Report [1]');
    expect(result.current.citations).toHaveLength(1);
    expect(result.current.isLoading).toBe(false);
  });

  it('defaults mode to auto and forwards an explicit mode', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('a'));
    await act(() => result.current.executeResearch('b', 'research'));
    expect(executeResearch.mock.calls[0][0].mode).toBe('auto');
    expect(executeResearch.mock.calls[1][0].mode).toBe('research');
  });

  it('reuses the thread id and sends prior messages as context', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('first'));
    await act(() => result.current.executeResearch('second'));
    const second = executeResearch.mock.calls[1][0];
    expect(second.thread_id).toBe('thread-1');
    expect(second.messages).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'Report [1]' },
    ]);
  });

  it('ignores blank queries', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('   '));
    expect(executeResearch).not.toHaveBeenCalled();
  });

  it('prefers the custom model from saved settings', async () => {
    localStorage.setItem(
      'mara_settings',
      JSON.stringify({ customModel: 'custom-x', defaultModel: 'default-y', apiKey: 'k' })
    );
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    expect(executeResearch.mock.calls[0][0]).toMatchObject({ model: 'custom-x', api_key: 'k' });
  });

  it('survives corrupt saved settings', async () => {
    localStorage.setItem('mara_settings', '{not json');
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    expect(executeResearch).toHaveBeenCalled();
  });

  it('records an error and an apology message on failure', async () => {
    executeResearch.mockRejectedValueOnce(new Error('boom'));
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    expect(result.current.error).toBe('boom');
    expect(result.current.messages[1].content).toContain('Error: boom');
    expect(result.current.isLoading).toBe(false);
  });

  it('flags reports that are awaiting human review', async () => {
    executeResearch.mockResolvedValueOnce({ ...RESPONSE, status: 'awaiting_review' });
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    expect(result.current.awaitingReview).toBe(true);
  });
});

describe('submitFeedback', () => {
  it('does nothing without an active thread', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.submitFeedback('x', false));
    expect(submitFeedback).not.toHaveBeenCalled();
  });

  it('sends feedback for the current thread and appends the revised report', async () => {
    submitFeedback.mockResolvedValue({ ...RESPONSE, report: 'Revised' });
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    await act(() => result.current.submitFeedback('shorter', false));
    expect(submitFeedback).toHaveBeenCalledWith('thread-1', { feedback: 'shorter', approved: false });
    expect(result.current.messages.at(-1)?.content).toBe('Revised');
  });
});

describe('clearMessages', () => {
  it('resets conversation state', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    act(() => result.current.clearMessages());
    expect(result.current.messages).toEqual([]);
    expect(result.current.citations).toEqual([]);
    expect(result.current.currentThreadId).toBeNull();
  });
});
