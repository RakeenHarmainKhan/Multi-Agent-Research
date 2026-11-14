/**
 * Settings: shows how the backend is configured. Model, keys and pricing are
 * server-side settings (backend/.env), so nothing here is editable.
 */
import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import type { HealthStatus } from '../types';

export const Settings: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    apiService
      .checkHealth()
      .then(setHealth)
      .catch(() => setError(true));
  }, []);

  return (
    <div className="h-full overflow-y-auto bg-slate-950 p-8">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-slate-100">Settings</h2>
        <p className="text-slate-400 mt-2">
          Configuration of the research backend
        </p>
      </div>

      <div className="max-w-4xl space-y-6">
        <Section title="Backend Configuration">
          <div className="space-y-3 text-sm">
            <Row
              label="Backend"
              value={error ? 'Unreachable' : health ? `Online · v${health.version}` : 'Checking…'}
            />
            <Row
              label="LLM"
              value={
                health
                  ? health.llm_service === 'configured'
                    ? 'Configured'
                    : 'Not configured'
                  : '—'
              }
            />
            <Row label="Run storage" value={health?.database ?? '—'} />
          </div>
          <p className="text-xs text-slate-500 mt-4 leading-relaxed">
            The model, API keys and optional pricing are set on the server in{' '}
            <code className="text-cyan-400">backend/.env</code> (see{' '}
            <code className="text-cyan-400">backend/.env.example</code>): the{' '}
            <code className="text-cyan-400">AZURE_OPENAI_*</code> variables select your deployment,{' '}
            <code className="text-cyan-400">TAVILY_API_KEY</code> enables web search, and{' '}
            <code className="text-cyan-400">LLM_*_COST_PER_1M_TOKENS</code> enables cost estimates.
            Keys are never sent from the browser.
          </p>
        </Section>

        {/* About */}
        <Section title="About MARA">
          <div className="space-y-3 text-sm text-slate-400">
            <p>
              <strong className="text-slate-300">Version:</strong> 1.0.0
            </p>
            <p>
              <strong className="text-slate-300">Framework:</strong> LangGraph +
              FastAPI + React
            </p>
            <p>
              <strong className="text-slate-300">License:</strong> MIT
            </p>
          </div>
        </Section>
      </div>
    </div>
  );
};

// Helper Components
const Section: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
    <h3 className="text-xl font-semibold text-slate-200 mb-4">{title}</h3>
    {children}
  </div>
);

const Row: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
    <span className="text-slate-300">{label}</span>
    <span className="text-slate-400">{value}</span>
  </div>
);
