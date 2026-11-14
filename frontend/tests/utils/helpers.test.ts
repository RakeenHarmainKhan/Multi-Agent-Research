import { describe, expect, it, vi } from 'vitest';
import {
  copyToClipboard,
  downloadTextFile,
  formatCost,
  formatDuration,
  generateThreadId,
  truncateText,
} from '../../src/utils/helpers';

describe('formatDuration', () => {
  it('shows milliseconds under one second', () => {
    expect(formatDuration(0.5)).toBe('500ms');
  });
  it('shows seconds under a minute', () => {
    expect(formatDuration(23.14)).toBe('23.1s');
  });
  it('shows minutes and seconds', () => {
    expect(formatDuration(125)).toBe('2m 5s');
  });
});

describe('formatCost', () => {
  it('formats as USD', () => {
    expect(formatCost(1.5)).toBe('$1.50');
  });
});

describe('truncateText', () => {
  it('leaves short text alone', () => {
    expect(truncateText('abc', 5)).toBe('abc');
  });
  it('adds an ellipsis when truncating', () => {
    expect(truncateText('abcdef', 3)).toBe('abc...');
  });
});

describe('generateThreadId', () => {
  it('is prefixed and unique', () => {
    const a = generateThreadId();
    const b = generateThreadId();
    expect(a.startsWith('thread_')).toBe(true);
    expect(a).not.toBe(b);
  });
});

describe('copyToClipboard', () => {
  it('returns true on success', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    expect(await copyToClipboard('hi')).toBe(true);
    expect(writeText).toHaveBeenCalledWith('hi');
  });

  it('returns false when the clipboard rejects', async () => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await copyToClipboard('hi')).toBe(false);
  });
});

describe('downloadTextFile', () => {
  it('creates a download link with the given filename and cleans up', () => {
    const createObjectURL = vi.fn().mockReturnValue('blob:x');
    const revokeObjectURL = vi.fn();
    Object.assign(window.URL, { createObjectURL, revokeObjectURL });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    downloadTextFile('# report', 'report.md');

    expect(createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:x');
    expect(document.querySelector('a[download]')).toBeNull();
  });
});
