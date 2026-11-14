/**
 * Renders report markdown: GFM tables/lists, KaTeX math and clickable [n] citations.
 */
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownRendererProps {
  content: string;
  citationIds?: number[];
}

/** Normalise LaTeX delimiters and turn valid [n] markers into #cite-n links. */
const prepareContent = (content: string, citationIds: number[]): string => {
  const ids = new Set(citationIds);
  return content
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => `$$${math}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => `$${math}$`)
    .replace(/\[(\d+)\](?!\()/g, (match, n) =>
      ids.has(Number(n)) ? `[${n}](#cite-${n})` : match
    );
};

const scrollToCitation = (event: React.MouseEvent, href: string) => {
  event.preventDefault();
  const target = document.getElementById(href.slice(1));
  target?.closest('details')?.setAttribute('open', '');
  target?.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  citationIds = [],
}) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      className="prose prose-invert prose-slate max-w-none"
      components={{
        h1: ({ children }) => (
          <h1 className="text-2xl font-bold text-slate-100 mt-6 mb-4">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-semibold text-slate-200 mt-5 mb-3">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-semibold text-slate-300 mt-4 mb-2">
            {children}
          </h3>
        ),
        p: ({ children }) => (
          <p className="text-slate-300 leading-relaxed mb-4">{children}</p>
        ),
        a: ({ href, children }) =>
          href?.startsWith('#cite-') ? (
            <sup>
              <a
                href={href}
                onClick={(event) => scrollToCitation(event, href)}
                className="text-blue-400 hover:text-blue-300 font-semibold no-underline px-0.5"
              >
                [{children}]
              </a>
            </sup>
          ) : (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 underline"
            >
              {children}
            </a>
          ),
        // react-markdown v9 has no `inline` prop: fenced blocks carry a language class or span lines
        code: ({ className, children }) =>
          /language-/.test(className ?? '') || String(children).includes('\n') ? (
            <code className="block bg-slate-800 text-slate-300 p-4 rounded-lg overflow-x-auto text-sm">
              {children}
            </code>
          ) : (
            <code className="bg-slate-800 text-cyan-400 px-1.5 py-0.5 rounded text-sm">
              {children}
            </code>
          ),
        ul: ({ children }) => (
          <ul className="list-disc pl-6 space-y-2 mb-4 text-slate-300 marker:text-slate-500">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-6 space-y-2 mb-4 text-slate-300 marker:text-slate-500">
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="pl-1 [&>p]:mb-1 [&>ul]:mt-2 [&>ol]:mt-2 [&>ul]:mb-0 [&>ol]:mb-0">
            {children}
          </li>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-500 pl-4 italic text-slate-400 my-4">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full border border-slate-700">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-slate-700 bg-slate-800 px-4 py-2 text-left font-semibold">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-slate-700 px-4 py-2">{children}</td>
        ),
      }}
    >
      {prepareContent(content, citationIds)}
    </ReactMarkdown>
  );
};