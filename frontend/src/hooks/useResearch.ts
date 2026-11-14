/**
 * Centralized state management
 * Custom React hook for research operations.
 * Makes api call to /research, 
 * Manages state and API calls for research workflow.
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
  awaitingReview: boolean;
  executeResearch: (query: string, mode?: ResearchMode) => Promise<void>;
  submitFeedback: (feedback: string, approved: boolean) => Promise<void>;
  clearMessages: () => void;
}

export const useResearch = (): UseResearchReturn => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [awaitingReview, setAwaitingReview] = useState(false);

  const executeResearch = useCallback(async (query: string, mode: ResearchMode = 'auto') => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: query,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const request: ResearchRequest = {
        query,
        mode,
        thread_id: currentThreadId || undefined,
        messages: messages.slice(-HISTORY_TO_SEND).map((message) => ({
          role: message.role,
          content: message.content,
        })),
      };

      const response = await apiService.executeResearch(request);

      // Store thread ID for conversation continuity
      setCurrentThreadId(response.thread_id);
      setCitations(response.citations || []);
      setAwaitingReview(response.status === 'awaiting_review');

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.report || 'Research completed but no report generated.',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Research failed';
      setError(errorMessage);

      const errorResponse: Message = {
        role: 'assistant',
        content: `Error: ${errorMessage}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  }, [currentThreadId, messages]);

  const submitFeedback = useCallback(async (feedback: string, approved: boolean) => {
    if (!currentThreadId) return;
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.submitFeedback(currentThreadId, { feedback, approved });
      setCurrentThreadId(response.thread_id);
      setCitations(response.citations || []);
      setAwaitingReview(response.status === 'awaiting_review');

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.report || 'Research updated.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Feedback failed');
    } finally {
      setIsLoading(false);
    }
  }, [currentThreadId]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setCurrentThreadId(null);
    setCitations([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    currentThreadId,
    citations,
    awaitingReview,
    executeResearch,
    submitFeedback,
    clearMessages,
  };
};