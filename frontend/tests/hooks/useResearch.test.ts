import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useResearch } from '../../src/hooks/useResearch';
import { apiService } from '../../src/services/api';

vi.mock('../../src/services/api', () => ({
  apiService: { executeResearch: vi.fn() },
}));

const executeResearch = vi.mocked(apiService.executeResearch);

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

  it('sends at most the last 10 messages as history', async () => {
    const { result } = renderHook(() => useResearch());
    for (let i = 0; i < 7; i++) {
      await act(() => result.current.executeResearch(`q${i}`));
    }
    const lastCall = executeResearch.mock.calls.at(-1)![0];
    expect(lastCall.messages).toHaveLength(10);
    expect(lastCall.messages!.at(-1)).toEqual({ role: 'assistant', content: 'Report [1]' });
  });

  it('ignores blank queries', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('   '));
    expect(executeResearch).not.toHaveBeenCalled();
  });

  it('never sends a model or API key from the browser', async () => {
    localStorage.setItem(
      'mara_settings',
      JSON.stringify({ customModel: 'custom-x', defaultModel: 'default-y', apiKey: 'k' })
    );
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    const sent = executeResearch.mock.calls[0][0] as unknown as Record<string, unknown>;
    expect(sent).not.toHaveProperty('model');
    expect(sent).not.toHaveProperty('api_key');
  });

  it('records an error and an apology message on failure', async () => {
    executeResearch.mockRejectedValueOnce(new Error('boom'));
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('q'));
    expect(result.current.error).toBe('boom');
    expect(result.current.messages[1].content).toContain('Error: boom');
    expect(result.current.isLoading).toBe(false);
  });

});

describe('regenerate', () => {
  it('re-asks the last question and replaces its answer', async () => {
    executeResearch
      .mockResolvedValueOnce({ ...RESPONSE, report: 'answer 1' })
      .mockResolvedValueOnce({ ...RESPONSE, report: 'answer 2' })
      .mockResolvedValueOnce({ ...RESPONSE, report: 'answer 2 (regenerated)' });
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.executeResearch('first'));
    await act(() => result.current.executeResearch('second'));
    await act(() => result.current.regenerate('research'));

    const regenerated = executeResearch.mock.calls[2][0];
    expect(regenerated.query).toBe('second'); // not "Regenerate previous response"
    expect(regenerated.mode).toBe('research');
    // Context excludes the question being regenerated and its old answer
    expect(regenerated.messages).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'answer 1' },
    ]);
    expect(result.current.messages.map((m) => m.content)).toEqual([
      'first',
      'answer 1',
      'second',
      'answer 2 (regenerated)',
    ]);
  });

  it('does nothing when there is no question yet', async () => {
    const { result } = renderHook(() => useResearch());
    await act(() => result.current.regenerate());
    expect(executeResearch).not.toHaveBeenCalled();
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
