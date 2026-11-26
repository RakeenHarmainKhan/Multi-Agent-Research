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
    <div className="h-full overflow-y-auto bg-slate-950 p-4 sm:p-8">
      <div className="mx-auto w-full max-w-3xl">
        <div className="mb-8">
          <h2 className="text-2xl font-medium text-slate-100 sm:text-3xl">Settings</h2>
          <p className="mt-2 text-slate-500">
            Backend connection and runtime details
          </p>
        </div>

        <div className="space-y-8">
          <Section title="Backend configuration">
            <dl className="divide-y divide-slate-700 border-y border-slate-700 text-sm">
              <Row
                label="Backend"
                value={error ? 'Unreachable' : health ? `Online · v${health.version}` : 'Checking'}
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
            </dl>
            <p className="mt-4 max-w-prose text-sm leading-relaxed text-slate-500">
              The model, API keys and optional pricing are set on the server in{' '}
              <code className="font-mono text-blue-400">backend/.env</code> (see{' '}
              <code className="font-mono text-blue-400">backend/.env.example</code>): the{' '}
              <code className="font-mono text-blue-400">AZURE_OPENAI_*</code> variables select your deployment,{' '}
              <code className="font-mono text-blue-400">TAVILY_API_KEY</code> enables web search, and{' '}
              <code className="font-mono text-blue-400">LLM_*_COST_PER_1M_TOKENS</code> enables cost estimates.
              Keys are never sent from the browser.
            </p>
          </Section>

          <Section title="About MARA">
            <dl className="divide-y divide-slate-700 border-y border-slate-700 text-sm">
              <Row label="Version" value="1.0.0" />
              <Row label="Framework" value="LangGraph + FastAPI + React" />
              <Row label="License" value="MIT" />
            </dl>
          </Section>
        </div>
      </div>
    </div>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <section>
    <h3 className="mb-3 text-lg font-medium text-slate-200">{title}</h3>
    {children}
  </section>
);

const Row: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-start justify-between gap-4 py-3">
    <dt className="shrink-0 text-slate-300">{label}</dt>
    <dd className="min-w-0 text-right text-slate-500">{value}</dd>
  </div>
);
