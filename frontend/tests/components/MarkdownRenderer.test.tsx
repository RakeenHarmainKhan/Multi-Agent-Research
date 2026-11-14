import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MarkdownRenderer } from '../../src/components/MarkdownRenderer';

const renderMd = (content: string, citationIds: number[] = []) =>
  render(<MarkdownRenderer content={content} citationIds={citationIds} />);

describe('lists', () => {
  it('renders bullets outside the text, not inline (regression: bullet on its own line)', () => {
    const { container } = renderMd('- first\n\n- second');
    const ul = container.querySelector('ul')!;
    expect(ul).toHaveClass('list-disc');
    expect(ul).not.toHaveClass('list-inside');
    expect(container.querySelectorAll('li')).toHaveLength(2);
  });

  it('keeps bold lead-in and text in the same list item', () => {
    const { container } = renderMd('- **Cost:** about $5 per run\n\n- **Speed:** fast');
    const first = container.querySelector('li')!;
    expect(first).toHaveTextContent('Cost: about');
    expect(first.querySelector('strong')).toHaveTextContent('Cost:');
  });

  it('renders ordered lists outside as well', () => {
    const { container } = renderMd('1. one\n2. two');
    const ol = container.querySelector('ol')!;
    expect(ol).toHaveClass('list-decimal');
    expect(ol).not.toHaveClass('list-inside');
  });

  it('renders nested lists', () => {
    const { container } = renderMd('- parent\n  - child');
    expect(container.querySelectorAll('ul')).toHaveLength(2);
  });
});

describe('markdown basics', () => {
  it('renders headings', () => {
    renderMd('## Executive Summary');
    expect(screen.getByRole('heading', { level: 2, name: 'Executive Summary' })).toBeInTheDocument();
  });

  it('opens normal links in a new tab', () => {
    renderMd('[site](https://example.com)');
    const link = screen.getByRole('link', { name: 'site' });
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('renders GFM tables', () => {
    const { container } = renderMd('| a | b |\n|---|---|\n| 1 | 2 |');
    expect(container.querySelector('table')).toBeInTheDocument();
  });
});

describe('code', () => {
  it('renders inline code inside the sentence, not as a block (regression)', () => {
    const { container } = renderMd('Use the `max_tokens` setting here.');
    const code = container.querySelector('p code')!;
    expect(code).toHaveTextContent('max_tokens');
    expect(code).not.toHaveClass('block');
  });

  it('renders fenced code blocks as blocks', () => {
    const { container } = renderMd('```python\nprint("hi")\n```');
    expect(container.querySelector('code')).toHaveClass('block');
  });

  it('renders fenced code without a language as a block', () => {
    const { container } = renderMd('```\nline one\nline two\n```');
    expect(container.querySelector('code')).toHaveClass('block');
  });
});

describe('math', () => {
  it('renders inline $...$ with KaTeX', () => {
    const { container } = renderMd('Energy is $E = mc^2$ here');
    expect(container.querySelector('.katex')).toBeInTheDocument();
  });

  it('renders display $$...$$ with KaTeX', () => {
    const { container } = renderMd('$$\nW = W_0 + BA\n$$');
    expect(container.querySelector('.katex-display')).toBeInTheDocument();
  });

  it('converts \\( \\) delimiters from the model', () => {
    const { container } = renderMd('Weight \\( W_0 \\) is frozen');
    expect(container.querySelector('.katex')).toBeInTheDocument();
  });

  it('converts \\[ \\] delimiters from the model', () => {
    const { container } = renderMd('\\[\n\\Delta W = AB\n\\]');
    expect(container.querySelector('.katex-display')).toBeInTheDocument();
  });

  it('keeps escaped currency as plain text', () => {
    const { container } = renderMd('It costs \\$5 to \\$10.');
    expect(container.querySelector('.katex')).not.toBeInTheDocument();
    expect(container).toHaveTextContent('It costs $5 to $10.');
  });
});

describe('citations', () => {
  it('turns a valid [n] marker into a superscript link', () => {
    const { container } = renderMd('Fact [1].', [1]);
    const link = container.querySelector('sup a')!;
    expect(link).toHaveAttribute('href', '#cite-1');
    expect(link).toHaveTextContent('[1]');
  });

  it('leaves markers without a matching source as plain text', () => {
    const { container } = renderMd('Fact [2].', [1]);
    expect(container.querySelector('sup a')).toBeNull();
    expect(container).toHaveTextContent('Fact [2].');
  });

  it('does nothing when there are no citations', () => {
    const { container } = renderMd('Fact [1].');
    expect(container.querySelector('sup a')).toBeNull();
  });

  it('does not rewrite [n](url) markdown links', () => {
    renderMd('[1](https://example.com)', [1]);
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com');
  });

  it('clicking a marker opens the sources panel and scrolls to the source', () => {
    document.body.insertAdjacentHTML(
      'beforeend',
      '<details id="panel"><div id="cite-1">source</div></details>'
    );
    const scroll = vi.fn();
    document.getElementById('cite-1')!.scrollIntoView = scroll;

    const { container } = renderMd('Fact [1].', [1]);
    fireEvent.click(container.querySelector('sup a')!);

    expect(document.getElementById('panel')).toHaveAttribute('open');
    expect(scroll).toHaveBeenCalled();
    document.getElementById('panel')!.remove();
  });
});
