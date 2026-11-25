/**
 * Main layout component with navigation sidebar.
 */
import React from 'react';
import type { TabType } from '../types';

interface LayoutProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({
  activeTab,
  onTabChange,
  children,
}) => {
  const tabs: Array<{ id: TabType; label: string }> = [
    { id: 'chat', label: 'Research Chat' },
    { id: 'history', label: 'History' },
    { id: 'dashboard', label: 'Analytics' },
    { id: 'settings', label: 'Settings' },
  ];

  return (
    <div className="flex h-dvh flex-col md:flex-row bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside className="w-full md:w-60 shrink-0 bg-slate-50 border-b md:border-b-0 md:border-r border-slate-800 flex flex-col">
        {/* Logo */}
        <div className="px-5 pt-3 pb-1 md:px-6 md:py-7 md:border-b border-slate-800">
          <h1 className="text-xl font-semibold tracking-wide text-slate-100">
            MARA
          </h1>
          <p className="hidden md:block text-xs text-slate-500 mt-1">
            Multi-Agent Research Assistant
          </p>
        </div>

        {/* Navigation */}
        <nav aria-label="Main navigation" className="grid grid-cols-4 md:flex md:flex-1 md:flex-col gap-1 p-2 md:p-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              aria-current={activeTab === tab.id ? 'page' : undefined}
              className={`whitespace-nowrap md:w-full text-center md:text-left px-1 md:px-3 py-2.5 rounded-lg text-[11px] sm:text-sm transition-colors ${activeTab === tab.id
                  ? 'bg-slate-800 text-slate-100 font-semibold'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="hidden md:block p-6 border-t border-slate-800 text-xs text-slate-500">
          <p>Version 1.0.0</p>
          <p className="mt-1">Powered by LangGraph</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 min-h-0 min-w-0 overflow-hidden flex flex-col">{children}</main>
    </div>
  );
};