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
        <div className="text-slate-400">Loading dashboard...</div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-400">Failed to load statistics</div>
      </div>
    );
  }

  const successRate = stats.total_queries > 0
    ? Math.round((stats.successful_queries / stats.total_queries) * 100)
    : 0;

  return (
    <div className="h-full overflow-y-auto bg-slate-950 p-4 sm:p-8">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-slate-100">System Analytics</h2>
        <p className="text-slate-400 mt-2">
          Runtime metrics for the current backend instance
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Queries"
          value={stats.total_queries.toString()}
          subtitle={`${stats.queries_today} today`}
        />
        <MetricCard
          title="Success Rate"
          value={`${successRate}%`}
          subtitle={`${stats.failed_queries} failed · ${stats.blocked_queries} blocked`}
        />
        <MetricCard
          title="Avg Latency"
          value={formatDuration(stats.avg_latency)}
          subtitle="Per completed query"
        />
        <MetricCard
          title="Tokens Used"
          value={stats.total_tokens.toLocaleString('en-US')}
          subtitle={`${Math.round(stats.avg_tokens_per_query).toLocaleString('en-US')} per query`}
        />
      </div>

      {/* System Health */}
      <div className="mt-8 bg-slate-900 border border-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-slate-200 mb-4">
          System Health
        </h3>
        <div className="space-y-3">
          <HealthIndicator
            label="API Server"
            status={health ? 'healthy' : 'down'}
            text={health ? `Operational · v${health.version}` : 'Unreachable'}
          />
          <HealthIndicator
            label="LLM Service"
            status={health?.llm_service === 'configured' ? 'healthy' : 'down'}
            text={health?.llm_service === 'configured' ? 'Configured' : 'Not configured'}
          />
          <HealthIndicator
            label="Storage"
            status="degraded"
            text={`${health?.database ?? 'unknown'} (resets on restart)`}
          />
        </div>
      </div>
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  subtitle: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
      <h3 className="text-slate-400 text-sm mb-2">{title}</h3>
      <p className="text-3xl font-semibold text-slate-100">
        {value}
      </p>
      <p className="text-slate-500 text-xs mt-2">{subtitle}</p>
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
    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
      <span className="text-slate-300">{label}</span>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${statusColor[status]}`} />
        <span className="text-sm text-slate-400">{text}</span>
      </div>
    </div>
  );
};