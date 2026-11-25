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
          <h2 className="text-base font-semibold">Research Assistant</h2>
          <p className="text-sm text-slate-500">
            Ask anything: quick answers, cited facts or full research reports
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={clearMessages}
            className="px-3 py-2 border border-slate-700 text-slate-300 rounded-lg hover:bg-slate-800 text-sm whitespace-nowrap"
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-6 md:px-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-full text-center">
            <div className="max-w-lg">
              <h3 className="font-display text-3xl font-semibold tracking-tight mb-4 text-slate-200">
                What would you like to explore?
              </h3>
              <p className="text-slate-400 mb-8">
                Ask a question for a direct answer, a cited fact, or a research report.
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
                      ? 'bg-blue-600 text-white'
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
                              <summary className="text-xs font-semibold text-slate-500 cursor-pointer hover:text-slate-300 transition-colors uppercase tracking-wider flex items-center gap-2">
                                <svg aria-hidden="true" className="h-3 w-3 group-open:rotate-90 transition-transform" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 3 5 5-5 5" /></svg>
                                Sources ({message.citations?.length})
                              </summary>
                              <div className="mt-4 grid grid-cols-1 gap-3">
                                {message.citations?.map((cite) => (
                                  <div key={cite.id} id={`cite-${idx}-${cite.id}`} className="p-3 bg-slate-950/50 rounded border border-slate-800 text-xs">
                                    <p className="text-slate-300 break-all">
                                      <span className="text-blue-400 font-semibold mr-1">[{cite.id}]</span>
                                      {cite.url ? (
                                        <a href={cite.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                                          {cite.title || cite.url}
                                        </a>
                                      ) : (
                                        cite.title
                                      )}
                                    </p>
                                    <p className="text-slate-500 mt-1 line-clamp-3">{cite.snippet}</p>
                                    <p className="text-slate-600 mt-1 uppercase text-[10px]">{cite.type}</p>
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
              <div className="flex flex-col space-y-4">
                <div className="flex items-center gap-3 p-5 bg-slate-900 rounded-lg border border-slate-800 max-w-md">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.2s' }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.4s' }}
                    ></div>
                  </div>
                  <span className="text-slate-400 text-sm">
                    Working on it...
                  </span>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-500 rounded-lg text-red-800 text-sm">
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
        <div className="flex gap-3 mb-2">
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Mode:
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as ResearchMode)}
              className="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-blue-500"
            >
              <option value="auto">Auto (Classify)</option>
              <option value="research">Web Research</option>
            </select>
          </label>
        </div>
        <div className="flex gap-3">
          <textarea
            aria-label="Research question"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask a research question on any topic... (Shift+Enter for new line)"
            className="min-w-0 flex-1 bg-slate-900 border border-slate-700 rounded-xl p-4 text-slate-200 placeholder-slate-500 focus:border-blue-500 resize-none"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-5 md:px-7 bg-blue-600 text-white rounded-xl hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          Research queries may take a little longer to complete.
        </p>
        </div>
      </div>
    </div>
  );
};
