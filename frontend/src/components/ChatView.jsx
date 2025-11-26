import { Bot, User, CornerDownLeft, LoaderCircle, FileText } from 'lucide-react';
import { useEffect, useRef } from 'react';

export default function ChatView({
  messages,
  loading,
  query,
  onQueryChange,
  onAsk,
  isProcessingComplete,
  onShowSources
}) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex-1 flex flex-col p-6 h-full">
      <div className="flex-1 overflow-y-auto mb-4 space-y-6 pr-4">
        {messages.length === 0 && !loading && (
          <div className="text-center text-slate-500 h-full flex flex-col items-center justify-center">
            <Bot size={48} className="mb-4" />
            <h2 className="text-2xl font-semibold">AI-Powered PDF Assistant</h2>
            <p className="mt-2">{isProcessingComplete ? "Ask a question to get started." : "Upload and process documents to begin."}</p>
          </div>
        )}
        {messages.map((msg, index) => (
          <div key={index} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'bot' && <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0"><Bot size={20} /></div>}
            <div className={`p-4 rounded-lg max-w-2xl shadow-sm ${msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white'}`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.role === 'bot' && msg.sources && msg.sources.length > 0 && (
                <button
                  onClick={() => onShowSources(index)}
                  className="mt-3 text-xs flex items-center gap-1 font-semibold hover:underline transition-colors_ text-blue-600"
                >
                  <FileText size={14} /> Show Sources
                </button>
              )}
            </div>
            {msg.role === 'user' && <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center flex-shrink-0"><User size={20} /></div>}
          </div>
        ))}
        {loading && (
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0"><Bot size={20} /></div>
            <div className="p-4 rounded-lg bg-white shadow-sm flex items-center"><LoaderCircle className="animate-spin" size={20} /></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={onAsk} className="relative">
        <textarea
          value={query}
          onChange={onQueryChange}
          placeholder={isProcessingComplete ? "Ask a question about your documents..." : "Process documents to begin."}
          disabled={!isProcessingComplete || loading}
          className="w-full p-4 pr-16 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none disabled:bg-slate-200"
          rows={1}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onAsk(e); } }}
        />
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:bg-slate-400 transition-colors"
        >
          <CornerDownLeft size={20} />
        </button>
      </form>
    </div>
  );
}