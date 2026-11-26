/**
 * Displays historical research queries with filtering and search.
 */
import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { formatDate, formatDuration, truncateText } from '../utils/helpers';
import { MarkdownRenderer } from './MarkdownRenderer';
import type { QueryHistory } from '../types';

interface ExtendedQueryHistory extends QueryHistory {
  report?: string;
}

export const ResearchHistory: React.FC = () => {
  const [history, setHistory] = useState<QueryHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortOrder, setSortOrder] = useState<string>('recent');
  const [selectedReport, setSelectedReport] = useState<ExtendedQueryHistory | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await apiService.getQueryHistory(50);
      setHistory(data);
      setLoadError(false);
    } catch (error) {
      console.error('Failed to load history:', error);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  };

  const filteredHistory = history
    .filter((item) => {
      const matchesSearch = item.query
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const matchesStatus =
        filterStatus === 'all' || item.status === filterStatus;
      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      if (sortOrder === 'recent') {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
      if (sortOrder === 'oldest') {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      }
      if (sortOrder === 'fastest') {
        return (a.execution_time || 0) - (b.execution_time || 0);
      }
      if (sortOrder === 'slowest') {
        return (b.execution_time || 0) - (a.execution_time || 0);
      }
      return 0;
    });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'border-green-500/30 bg-green-500/10 text-green-800';
      case 'failed':
        return 'border-red-500/30 bg-red-500/10 text-red-800';
      default:
        return 'border-slate-700 text-slate-500';
    }
  };

  const formatLabel = (value: string) =>
    value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-500">Loading history</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-slate-950 p-4 sm:p-8">
      <div className="mx-auto w-full max-w-screen-2xl">
        <div className="mb-8">
          <h2 className="text-2xl font-medium text-slate-100 sm:text-3xl">Research history</h2>
          <p className="mt-2 text-slate-500">
            Review past questions, outcomes, and reports
          </p>
        </div>

        <div className="mb-6 border-b border-slate-700 pb-6">
          <div className="flex flex-col gap-3 md:flex-row">
            <div className="flex-1">
              <input
                aria-label="Search queries"
                type="text"
                placeholder="Search"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <select
                title="Status filter"
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-slate-200 focus:border-blue-500 focus:outline-none md:w-auto"
              >
                <option value="all">All statuses</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </div>

            <div>
              <select
                title="Sort order"
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-slate-200 focus:border-blue-500 focus:outline-none md:w-auto"
              >
                <option value="recent">Recent first</option>
                <option value="oldest">Oldest first</option>
                <option value="fastest">Fastest</option>
                <option value="slowest">Slowest</option>
              </select>
            </div>

            <button
              onClick={loadHistory}
              className="rounded-lg bg-blue-600 px-5 py-2 text-white transition-colors hover:bg-blue-300"
            >
              Refresh
            </button>
          </div>
        </div>

        {loadError && (
          <p role="alert" className="mb-4 rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
            Could not load research history. Check the backend connection and select Refresh to try again.
          </p>
        )}

        <p className="mb-4 text-sm text-slate-500">
          Showing {filteredHistory.length} of {history.length} queries
        </p>

        <div className="divide-y divide-slate-700 border-y border-slate-700">
          {filteredHistory.length === 0 && !loadError ? (
            <div className="py-12 text-center">
              <p className="text-slate-500">
                {searchTerm || filterStatus !== 'all'
                  ? 'No queries match your filters'
                  : 'No research history yet'}
              </p>
            </div>
          ) : (
            filteredHistory.map((item) => (
              <div key={item.id} className="py-5">
                <div className="mb-3 flex flex-col items-start justify-between gap-4 sm:flex-row">
                  <div className="min-w-0 flex-1">
                    <h3 className="mb-2 break-words text-base font-medium text-slate-200">
                      {truncateText(item.query, 150)}
                    </h3>
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                      <span
                        className={`rounded-md border px-2 py-1 text-xs font-medium ${getStatusColor(
                          item.status
                        )}`}
                      >
                        {formatLabel(item.status)}
                      </span>

                      {item.intent && (
                        <span className="rounded-md border border-slate-700 px-2 py-1 text-xs font-medium text-slate-500">
                          {formatLabel(item.intent)}
                        </span>
                      )}

                      {item.execution_time && (
                        <span className="text-xs tabular-nums text-slate-500">
                          {formatDuration(item.execution_time)}
                        </span>
                      )}

                      <span className="text-xs text-slate-500">
                        ID: {item.thread_id.substring(0, 12)}...
                      </span>
                    </div>
                  </div>

                  <button
                    onClick={async () => {
                      setLoading(true);
                      try {
                        const result = await apiService.getRun(item.run_id);
                        setSelectedReport({
                          ...item,
                          report: result.report
                        });
                      } catch (err) {
                        console.error("Failed to load report", err);
                        alert("Failed to load report content");
                      } finally {
                        setLoading(false);
                      }
                    }}
                    className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-800"
                  >
                    View report
                  </button>
                </div>

                <div className="text-xs tabular-nums text-slate-500">
                  {formatDate(item.created_at)}
                </div>
              </div>
            ))
          )}
        </div>

        {selectedReport && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-3 sm:p-4">
            <div className="flex max-h-[90vh] w-full max-w-5xl flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-lg">
              <div className="flex items-center justify-between gap-4 border-b border-slate-700 p-4 sm:p-6">
                <div className="min-w-0">
                  <h2 className="text-lg font-medium text-slate-100 sm:text-xl">Research report</h2>
                  <p className="mt-1 line-clamp-2 break-words text-sm text-slate-500">
                    {selectedReport.query}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedReport(null)}
                  aria-label="Close report"
                  className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-800"
                >
                  <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="m5 5 10 10M15 5 5 15" />
                  </svg>
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 sm:p-8">
                <MarkdownRenderer content={selectedReport.report || "No content found for this report."} />
              </div>
              <div className="flex flex-wrap justify-end gap-3 border-t border-slate-700 p-4 sm:p-6">
                <button
                  onClick={() => {
                    if (selectedReport.report) {
                      navigator.clipboard.writeText(selectedReport.report);
                      alert("Copied to clipboard");
                    }
                  }}
                  className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-800"
                >
                  Copy content
                </button>
                <button
                  onClick={() => setSelectedReport(null)}
                  className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-300"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};