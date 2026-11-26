/**
 * Main chat interface: query input, mode selection and research results display.
 * Sends the query with thread_id and recent messages; renders the markdown report,
 * and its cited sources.
 */
import React, { useState, useRef, useEffect } from 'react';
import { useResearch } from '../hooks/useResearch';
import { MarkdownRenderer } from './MarkdownRenderer';
import { downloadTextFile } from '../utils/helpers';
import type { ResearchMode } from '../types';

// What each backend agent is doing, shown while the answer is on its way.
const STAGE_LABELS: Record<string, string> = {
  input_guard: 'Checking the question',
  intent_classifier: 'Deciding how much research it needs',
  simple_chat: 'Writing the answer',
  planner: 'Planning searches',
  researcher: 'Searching sources',
  writer: 'Writing the cited answer',
};

export const ChatInterface: React.FC = () => {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<ResearchMode>('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const {
    messages,
    isLoading,
    error,
    executeResearch,
    regenerate,
    clearMessages,
    redactions,
    stage,
  } = useResearch();

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    const query = input.trim();
    setInput('');
    await executeResearch(query, mode);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 px-5 py-4 md:px-8 border-b border-slate-800">
        <div>
          <h2 className="text-base font-medium">Research assistant</h2>
          <p className="text-sm text-slate-500">
            Direct answers, cited lookups, and multi-source reports
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={clearMessages}
            className="px-3 py-2 border border-slate-700 text-slate-300 rounded-lg hover:bg-slate-800 text-sm whitespace-nowrap"
          >
            Clear chat
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-6 md:px-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-full text-center">
            <div className="max-w-lg">
              <h3 className="font-display mb-4 text-2xl font-medium text-slate-200 sm:text-3xl">
                Ask MARA a question
              </h3>
              <p className="mb-8 text-slate-500">
                Get a direct answer, a cited lookup, or a research report.
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, idx) => {
              const isLast = idx === messages.length - 1;
              return (
                <div
                  key={idx}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                >
                  <div
                    className={`max-w-4xl ${message.role === 'user'
                      ? 'border border-slate-700 bg-slate-800 text-slate-100'
                      : 'bg-slate-900 border border-slate-800'
                      } rounded-2xl p-5`}
                  >
                    {message.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                      <div>
                        <MarkdownRenderer
                          content={message.content}
                          citationIds={(message.citations ?? []).map((c) => c.id)}
                          citationPrefix={`cite-${idx}`}
                        />

                        {isLast && redactions.length > 0 && (
                          <p className="mt-4 text-xs text-slate-500">
                            Masked before processing: {redactions.join(', ')}.
                          </p>
                        )}

                        {/* Sources Expander */}
                        {(message.citations?.length ?? 0) > 0 && (
                          <div className="mt-6 border-t border-slate-800 pt-4">
                            <details className="group">
                              <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-slate-500 transition-colors hover:text-slate-300">
                                <svg aria-hidden="true" className="h-4 w-4 transition-transform group-open:rotate-90" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                                  <path d="m6 3 5 5-5 5" />
                                </svg>
                                Sources ({message.citations?.length})
                              </summary>
                              <div className="mt-4 divide-y divide-slate-700 border-y border-slate-700">
                                {message.citations?.map((cite) => (
                                  <div key={cite.id} id={`cite-${idx}-${cite.id}`} className="py-3 text-sm">
                                    <p className="break-all text-slate-300">
                                      <span className="mr-1 font-medium text-blue-400">[{cite.id}]</span>
                                      {cite.url ? (
                                        <a href={cite.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                                          {cite.title || cite.url}
                                        </a>
                                      ) : (
                                        cite.title
                                      )}
                                    </p>
                                    <p className="mt-1 line-clamp-3 text-slate-500">{cite.snippet}</p>
                                    <p className="mt-1 text-xs capitalize text-slate-500">{cite.type}</p>
                                  </div>
                                ))}
                              </div>
                            </details>
                          </div>
                        )}

                        <div className="flex gap-2 mt-4 pt-4 border-t border-slate-700/50">
                          <button
                            onClick={() => navigator.clipboard.writeText(message.content)}
                            className="px-3 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors"
                          >
                            Copy
                          </button>
                          <button
                            onClick={() => downloadTextFile(message.content, `MARA_Report_${new Date().getTime()}.md`)}
                            className="px-3 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors"
                          >
                            Download .md
                          </button>
                          {isLast && (
                            <button
                              onClick={() => regenerate(mode)}
                              disabled={isLoading}
                              className="px-3 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors disabled:opacity-50"
                            >
                              Regenerate
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Loading Indicator */}
            {isLoading && (
              <div
                role="status"
                aria-live="polite"
                className="flex max-w-md items-center gap-3 rounded-lg border border-slate-700 bg-slate-900 p-4"
              >
                <span aria-hidden="true" className="h-2 w-2 rounded-full bg-blue-400" />
                <span className="text-sm text-slate-500">
                  {(stage && STAGE_LABELS[stage]) || 'Preparing a response'}
                </span>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="px-5 pb-5 pt-3 md:px-8 bg-slate-950">
        <div className="max-w-4xl mx-auto">
          <div className="mb-2 flex gap-3">
            <label className="flex items-center gap-2 text-xs text-slate-400">
              Mode
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as ResearchMode)}
                className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
              >
                <option value="auto">Automatic</option>
                <option value="research">Web research</option>
              </select>
            </label>
          </div>
          <div className="flex gap-3">
            <textarea
              aria-label="Research question"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask a question"
              className="min-w-0 flex-1 resize-none rounded-xl border border-slate-700 bg-slate-900 p-4 text-slate-200 placeholder-slate-500 focus:border-blue-500"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="rounded-xl bg-blue-600 px-5 font-medium text-white transition-colors hover:bg-blue-300 disabled:cursor-not-allowed disabled:opacity-50 md:px-7"
            >
              Send
            </button>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Enter to send · Shift+Enter for a new line · Research requests can take longer
          </p>
        </div>
      </div>
    </div>
  );
};
