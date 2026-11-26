/**
 * Display system metrics & health checks
 */
import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { formatDuration } from '../utils/helpers';
import type { HealthStatus, SystemStats } from '../types';

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      const [statsResult, healthResult] = await Promise.allSettled([
        apiService.getSystemStats(),
        apiService.checkHealth(),
      ]);
      setStats(statsResult.status === 'fulfilled' ? statsResult.value : null);
      setHealth(healthResult.status === 'fulfilled' ? healthResult.value : null);
      if (statsResult.status === 'rejected') {
        console.error('Failed to load stats:', statsResult.reason);
      }
      setLoading(false);
    };
    load();
    const interval = setInterval(load, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-500">Loading dashboard</div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-500">Could not load statistics</div>
      </div>
    );
  }

  const successRate = stats.total_queries > 0
    ? Math.round((stats.successful_queries / stats.total_queries) * 100)
    : 0;

  return (
    <div className="h-full overflow-y-auto bg-slate-950 p-4 sm:p-8">
      <div className="mx-auto w-full max-w-screen-2xl">
        <div className="mb-8">
          <h2 className="text-2xl font-medium text-slate-100 sm:text-3xl">System analytics</h2>
          <p className="mt-2 text-slate-500">
            Runtime metrics for the current backend instance
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-4 rounded-lg border border-slate-700 bg-slate-900 p-4 sm:grid-cols-4 sm:gap-0 sm:p-2">
          <Metric
            title="Total queries"
            value={stats.total_queries.toLocaleString('en-US')}
            subtitle={`${stats.queries_today.toLocaleString('en-US')} today`}
          />
          <Metric
            title="Success rate"
            value={`${successRate}%`}
            subtitle={`${stats.failed_queries} failed · ${stats.blocked_queries} blocked`}
          />
          <Metric
            title="Average latency"
            value={formatDuration(stats.avg_latency)}
            subtitle="Per completed query"
          />
          <Metric
            title="Tokens used"
            value={stats.total_tokens.toLocaleString('en-US')}
            subtitle={`${Math.round(stats.avg_tokens_per_query).toLocaleString('en-US')} per query`}
          />
        </dl>

        <section className="mt-8">
          <h3 className="mb-3 text-lg font-medium text-slate-200">System health</h3>
          <dl className="divide-y divide-slate-700 border-y border-slate-700">
            <HealthIndicator
              label="API server"
              status={health ? 'healthy' : 'down'}
              text={health ? `Operational · v${health.version}` : 'Unreachable'}
            />
            <HealthIndicator
              label="LLM service"
              status={health?.llm_service === 'configured' ? 'healthy' : 'down'}
              text={health?.llm_service === 'configured' ? 'Configured' : 'Not configured'}
            />
            <HealthIndicator
              label="Storage"
              status="degraded"
              text={`${health?.database ?? 'unknown'} (resets on restart)`}
            />
          </dl>
        </section>
      </div>
    </div>
  );
};

interface MetricProps {
  title: string;
  value: string;
  subtitle: string;
}

const Metric: React.FC<MetricProps> = ({
  title,
  value,
  subtitle,
}) => {
  return (
    <div className="min-w-0 px-2 py-3 sm:px-4 sm:py-4">
      <dt className="text-sm font-medium text-slate-500">{title}</dt>
      <dd className="mt-2 break-words text-2xl font-medium tabular-nums text-slate-100 sm:text-3xl">
        {value}
      </dd>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
};

interface HealthIndicatorProps {
  label: string;
  status: 'healthy' | 'degraded' | 'down';
  text: string;
}

const HealthIndicator: React.FC<HealthIndicatorProps> = ({
  label,
  status,
  text,
}) => {
  const statusColor = {
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    down: 'bg-red-500',
  };

  return (
    <div className="flex flex-col gap-1 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <dt className="text-slate-300">{label}</dt>
      <dd className="flex items-center gap-2 text-sm text-slate-500">
        <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${statusColor[status]}`} />
        {text}
      </dd>
    </div>
  );
};