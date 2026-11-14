import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatInterface } from '../../src/components/ChatInterface';
import { apiService } from '../../src/services/api';

vi.mock('../../src/services/api', () => ({
  apiService: {
    executeResearch: vi.fn(),
  },
}));

const executeResearch = vi.mocked(apiService.executeResearch);

const RESPONSE = {
  thread_id: 't1',
  status: 'completed' as const,
  report: 'LoRA adds low-rank updates [1].',
  citations: [
    { id: 1, type: 'web' as const, title: 'Alpha Blog', url: 'http://alpha.test', snippet: 'alpha snippet' },
    { id: 2, type: 'academic' as const, title: 'Paper X', url: '', snippet: 'paper snippet' },
  ],
  sources: [],
};

const send = async (text: string) => {
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/Ask a research question/), text);
  await user.click(screen.getByRole('button', { name: 'Send' }));
  return user;
};

beforeEach(() => {
  executeResearch.mockResolvedValue(RESPONSE);
});

describe('mode dropdown', () => {
  it('defaults to Auto', () => {
    render(<ChatInterface />);
    expect(screen.getByRole('combobox')).toHaveValue('auto');
  });

  it('offers Auto and Web Research', () => {
    render(<ChatInterface />);
    const options = screen.getAllByRole('option').map((o) => o.textContent);
    expect(options).toEqual(['Auto (Classify)', 'Web Research']);
  });

  it('sends mode "auto" by default', async () => {
    render(<ChatInterface />);
    await send('what is lora');
    await waitFor(() => expect(executeResearch).toHaveBeenCalled());
    expect(executeResearch.mock.calls[0][0]).toMatchObject({ query: 'what is lora', mode: 'auto' });
  });

  it('sends mode "research" after selecting Web Research', async () => {
    render(<ChatInterface />);
    const user = userEvent.setup();
    await user.selectOptions(screen.getByRole('combobox'), 'research');
    await user.type(screen.getByPlaceholderText(/Ask a research question/), 'what is lora');
    await user.click(screen.getByRole('button', { name: 'Send' }));
    await waitFor(() => expect(executeResearch).toHaveBeenCalled());
    expect(executeResearch.mock.calls[0][0]).toMatchObject({ mode: 'research' });
  });
});

describe('sending', () => {
  it('disables Send until there is text', async () => {
    render(<ChatInterface />);
    const button = screen.getByRole('button', { name: 'Send' });
    expect(button).toBeDisabled();
    await userEvent.setup().type(screen.getByPlaceholderText(/Ask a research question/), 'hi');
    expect(button).toBeEnabled();
  });

  it('shows the user message and the assistant report', async () => {
    render(<ChatInterface />);
    await send('what is lora');
    expect(await screen.findByText(/LoRA adds low-rank updates/)).toBeInTheDocument();
    expect(screen.getByText('what is lora')).toBeInTheDocument();
  });

  it('shows an error message when the request fails', async () => {
    executeResearch.mockRejectedValueOnce(new Error('Network Error'));
    render(<ChatInterface />);
    await send('what is lora');
    expect((await screen.findAllByText(/Network Error/)).length).toBeGreaterThan(0);
  });

  it('Regenerate re-sends the original question, not placeholder text', async () => {
    render(<ChatInterface />);
    const user = await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    await user.click(screen.getByRole('button', { name: 'Regenerate' }));
    await waitFor(() => expect(executeResearch).toHaveBeenCalledTimes(2));
    expect(executeResearch.mock.calls[1][0]).toMatchObject({ query: 'what is lora', mode: 'auto' });
    expect(screen.getAllByText('what is lora')).toHaveLength(1);
  });

  it('shows one honest loading message while waiting', async () => {
    let finish: (value: typeof RESPONSE) => void = () => {};
    executeResearch.mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    render(<ChatInterface />);
    await send('what is lora');
    expect(await screen.findByText('Working on it...')).toBeInTheDocument();
    finish(RESPONSE);
    await screen.findByText(/LoRA adds low-rank updates/);
    expect(screen.queryByText('Working on it...')).not.toBeInTheDocument();
  });

  it('clears the conversation', async () => {
    render(<ChatInterface />);
    const user = await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    await user.click(screen.getByRole('button', { name: 'Clear Chat' }));
    expect(screen.queryByText(/LoRA adds low-rank updates/)).not.toBeInTheDocument();
    expect(screen.getByText(/Welcome to MARA/)).toBeInTheDocument();
  });
});

describe('guardrails', () => {
  it('tells the user what was masked before processing', async () => {
    executeResearch.mockResolvedValueOnce({
      ...RESPONSE,
      guardrails: { blocked: false, reasons: [], redactions: ['email', 'phone'] },
    });
    render(<ChatInterface />);
    await send('email me at ana@corp.com');
    expect(await screen.findByText('Masked before processing: email, phone.')).toBeInTheDocument();
  });

  it('shows no notice when nothing was masked', async () => {
    executeResearch.mockResolvedValueOnce({
      ...RESPONSE,
      guardrails: { blocked: false, reasons: [], redactions: [] },
    });
    render(<ChatInterface />);
    await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    expect(screen.queryByText(/Masked before processing/)).not.toBeInTheDocument();
  });

  it('shows a blocked answer as a normal assistant message', async () => {
    executeResearch.mockResolvedValueOnce({
      ...RESPONSE,
      status: 'blocked',
      report: "I can't help with that request.",
      citations: [],
      guardrails: { blocked: true, reasons: ['unsafe'], redactions: [] },
    });
    render(<ChatInterface />);
    await send('something harmful');
    expect(await screen.findByText("I can't help with that request.")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Regenerate' })).toBeInTheDocument();
  });

  it('keeps the user their original text on screen', async () => {
    executeResearch.mockResolvedValueOnce({
      ...RESPONSE,
      guardrails: { blocked: false, reasons: [], redactions: ['email'] },
    });
    render(<ChatInterface />);
    await send('email me at ana@corp.com');
    await screen.findByText(/Masked before processing/);
    expect(screen.getByText('email me at ana@corp.com')).toBeInTheDocument();
  });
});

describe('sources panel', () => {
  it('lists cited sources with titles, links and snippets', async () => {
    render(<ChatInterface />);
    await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);

    expect(screen.getByText('Sources (2)')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: 'Alpha Blog' });
    expect(link).toHaveAttribute('href', 'http://alpha.test');
    expect(screen.getByText('alpha snippet')).toBeInTheDocument();
  });

  it('shows sources without a URL (arXiv) as plain titles', async () => {
    render(<ChatInterface />);
    await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    expect(screen.getByText('Paper X')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Paper X' })).not.toBeInTheDocument();
  });

  it('renders the [1] marker in the report as a link to the source card', async () => {
    const { container } = render(<ChatInterface />);
    await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    expect(container.querySelector('sup a[href="#cite-1"]')).toBeInTheDocument();
    expect(container.querySelector('#cite-1')).toBeInTheDocument();
  });

  it('hides the panel when the response has no citations', async () => {
    executeResearch.mockResolvedValueOnce({ ...RESPONSE, citations: [] });
    render(<ChatInterface />);
    await send('what is lora');
    await screen.findByText(/LoRA adds low-rank updates/);
    expect(screen.queryByText(/^Sources \(/)).not.toBeInTheDocument();
  });
});
