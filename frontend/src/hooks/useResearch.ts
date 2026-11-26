/**
 * Chat state and API calls for the research workflow (ask, regenerate).
 */
import { useState, useCallback } from 'react';
import { apiService } from '../services/api';
import type { Message, ResearchMode, ResearchRequest, Citation } from '../types';

// The backend only uses the last few turns as context; don't ship the whole chat.
const HISTORY_TO_SEND = 10;

interface UseResearchReturn {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  currentThreadId: string | null;
  citations: Citation[];
  /** Kinds of personal data masked in the latest question, e.g. ['email'] */
  redactions: string[];
  /** Name of the agent currently working on the question, while loading. */
  stage: string | null;
  executeResearch: (query: string, mode?: ResearchMode) => Promise<void>;
  /** Re-run the last question, replacing its answer. */
  regenerate: (mode?: ResearchMode) => Promise<void>;
  clearMessages: () => void;
}

export const useResearch = (): UseResearchReturn => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [redactions, setRedactions] = useState<string[]>([]);
  const [stage, setStage] = useState<string | null>(null);

  const appendAssistant = useCallback((content: string, answerCitations: Citation[] = []) => {
    setMessages((prev) => [...prev, { role: 'assistant', content, timestamp: new Date(), citations: answerCitations }]);
  }, []);

  /** Ask `query` with `history` as context; the chat becomes `history` + question + answer. */
  const runQuery = useCallback(async (query: string, mode: ResearchMode, history: Message[]) => {
    setIsLoading(true);
    setError(null);
    setMessages([...history, { role: 'user', content: query, timestamp: new Date() }]);

    try {
      const request: ResearchRequest = {
        query,
        mode,
        thread_id: currentThreadId || undefined,
        messages: history.slice(-HISTORY_TO_SEND).map(({ role, content }) => ({ role, content })),
      };

      const response = await apiService.executeResearch(request, setStage);

      // Store thread ID for conversation continuity
      setCurrentThreadId(response.thread_id);
      setCitations(response.citations || []);
      setRedactions(response.guardrails?.redactions ?? []);
      appendAssistant(response.report || 'Research completed but no report generated.', response.citations || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Research failed';
      setError(errorMessage);
      appendAssistant(`Error: ${errorMessage}. Please try again.`);
    } finally {
      setIsLoading(false);
      setStage(null);
    }
  }, [currentThreadId, appendAssistant]);

  const executeResearch = useCallback(async (query: string, mode: ResearchMode = 'auto') => {
    if (!query.trim()) return;
    await runQuery(query, mode, messages);
  }, [runQuery, messages]);

  const regenerate = useCallback(async (mode: ResearchMode = 'auto') => {
    let lastUserIndex = messages.length - 1;
    while (lastUserIndex >= 0 && messages[lastUserIndex].role !== 'user') lastUserIndex--;
    if (lastUserIndex === -1) return;
    // Drop the last question and its answer, then ask the same question again.
    await runQuery(messages[lastUserIndex].content, mode, messages.slice(0, lastUserIndex));
  }, [runQuery, messages]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentThreadId(null);
    setCitations([]);
    setRedactions([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    currentThreadId,
    citations,
    redactions,
    stage,
    executeResearch,
    regenerate,
    clearMessages,
  };
};