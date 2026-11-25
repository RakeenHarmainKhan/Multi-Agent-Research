/**
 * TypeScript type definitions for the MARA frontend.
 * Ensures type safety across components and API interactions.
 */

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

export interface Citation {
  id: number;
  type: 'web' | 'academic';
  title: string;
  url: string;
  snippet: string;
}

export interface Source {
  id: number;
  type: 'web' | 'academic';
  title?: string;
  url?: string;
  content: string;
  score?: number;
}

export type ResearchMode = 'auto' | 'research';

export interface ResearchRequest {
  query: string;
  thread_id?: string;
  mode?: ResearchMode;
  messages?: Array<Pick<Message, 'role' | 'content'>>;
}

export interface Guardrails {
  blocked: boolean;
  reasons: string[];
  /** Kinds of personal data masked before processing, e.g. ['email'] */
  redactions: string[];
}

export interface ResearchResponse {
  thread_id: string;
  status: 'processing' | 'completed' | 'failed' | 'blocked';
  report?: string;
  citations: Citation[];
  sources: Source[];
  execution_time?: number;
  input_tokens?: number;
  output_tokens?: number;
  estimated_cost?: number | null;
  guardrails?: Guardrails;
}

export interface SystemStats {
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  blocked_queries: number;
  avg_latency: number;
  total_tokens: number;
  avg_tokens_per_query: number;
  /** null when LLM pricing isn't configured on the backend */
  total_cost: number | null;
  queries_today: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  uptime: number;
  database: string;
  llm_service: 'configured' | 'not_configured';
  timestamp: string;
}

export interface QueryHistory {
  id: number;
  run_id: string;
  thread_id: string;
  query: string;
  status: string;
  intent?: string;
  execution_time?: number;
  created_at: string;
}



export type TabType = 'chat' | 'dashboard' | 'history' | 'settings';