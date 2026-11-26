/**
 * API service for MARA frontend - No authentication.
 */
import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  HealthStatus,
  QueryHistory,
  ResearchRequest,
  ResearchResponse,
  SystemStats,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/v1`,
      timeout: 300000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
      }
    );
  }

  /** Runs the workflow over SSE; `onStage` receives each agent's name as it starts. */
  async executeResearch(request: ResearchRequest, onStage?: (stage: string) => void): Promise<ResearchResponse> {
    const response = await fetch(`${API_BASE_URL}/api/v1/research/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(300000),
    });
    if (!response.ok || !response.body) throw new Error(`Request failed with status code ${response.status}`);

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() ?? '';
      for (const block of blocks) {
        const event = block.match(/^event: (.*)$/m)?.[1];
        const data = JSON.parse(block.match(/^data: (.*)$/m)?.[1] ?? 'null');
        if (event === 'stage') onStage?.(data.stage);
        else if (event === 'result') return data as ResearchResponse;
        else if (event === 'error') throw new Error(data.detail);
      }
    }
    throw new Error('The connection closed before the answer arrived');
  }

  async getResearchResult(threadId: string): Promise<ResearchResponse> {
    const response = await this.client.get<ResearchResponse>(`/research/${threadId}`);
    return response.data;
  }

  /** One specific question's result; a conversation thread can contain several runs. */
  async getRun(runId: string): Promise<ResearchResponse> {
    const response = await this.client.get<ResearchResponse>(`/runs/${runId}`);
    return response.data;
  }

  async getSystemStats(): Promise<SystemStats> {
    const response = await this.client.get<SystemStats>('/admin/stats');
    return response.data;
  }

  async getQueryHistory(limit: number = 20): Promise<QueryHistory[]> {
    const response = await this.client.get<QueryHistory[]>('/admin/history', {
      params: { limit },
    });
    return response.data;
  }

  async checkHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }
}

export const apiService = new ApiService();