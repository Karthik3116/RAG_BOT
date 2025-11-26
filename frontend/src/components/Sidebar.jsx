import { UploadCloud, LoaderCircle, Plus, MessageSquare, Trash2, FileText } from 'lucide-react';
import { useRef } from 'react';

// A helper component for displaying file status
const FileStatus = ({ files, processedFileNames }) => {
  // If new files have been selected for upload
  if (files.length > 0) {
    const fileCountText = files.length === 1 ? `1 file selected` : `${files.length} files selected`;
    const fileNames = files.map(f => f.name).join(', ');
    return (
      <div className="text-xs text-slate-600" title={fileNames}>
        <strong>{fileCountText}</strong>
        <p className="truncate">{fileNames}</p>
      </div>
    );
  }

  // If no new files are selected, show already processed files for the chat
  if (processedFileNames && processedFileNames.length > 0) {
    return (
      <div className="text-xs text-slate-500 space-y-1">
        <p className="font-semibold">Active in this chat:</p>
        <div className="flex items-center gap-1.5 truncate" title={processedFileNames.join(', ')}>
          <FileText size={14} className="flex-shrink-0" />
          <span className="truncate">{processedFileNames.join(', ')}</span>
        </div>
      </div>
    );
  }

  // Default message
  return <p className="text-xs text-slate-500">No files selected for this chat.</p>;
};


export default function Sidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  files,
  onFileChange,
  processing,
  onProcess,
  processedFileNames // New prop
}) {
  const fileInputRef = useRef(null);
  
  const isChatProcessed = processedFileNames && processedFileNames.length > 0;

  return (
    <aside className="w-80 bg-slate-50 border-r border-slate-200 p-4 flex flex-col h-screen">
      <div className="flex-1 flex flex-col overflow-hidden">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-white border border-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-md hover:bg-slate-100 transition-colors mb-4"
        >
          <Plus size={18} /> New Chat
        </button>
        <div className="flex-1 overflow-y-auto pr-2">
          <h2 className="text-sm font-semibold text-slate-500 mb-2 px-2">Chats</h2>
          <div className="space-y-1">
            {Object.entries(chats).map(([chatId, chat]) => (
              <div
                key={chatId}
                onClick={() => onSelectChat(chatId)}
                className={`group flex items-center justify-between p-2 rounded-md cursor-pointer ${activeChatId === chatId ? 'bg-blue-100 text-blue-700' : 'hover:bg-slate-200'}`}
              >
                <div className="flex items-center gap-2 truncate">
                  <MessageSquare size={16} />
                  <span className="truncate text-sm">{chat.title}</span>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); onDeleteChat(chatId); }}
                  className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-500 transition-opacity"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border-t border-slate-200 pt-4">
        <h2 className="text-lg font-bold mb-4">⚙️ Setup</h2>
        <div className="space-y-3">
          <div
            className="w-full h-28 border-2 border-dashed border-slate-300 rounded-md flex flex-col items-center justify-center text-slate-500 cursor-pointer hover:bg-slate-100 transition-colors"
            onClick={() => fileInputRef.current.click()}
          >
            <UploadCloud size={24} />
            <p className="text-sm mt-1">Upload PDF Files</p>
            <input type="file" multiple ref={fileInputRef} onChange={onFileChange} className="hidden" accept=".pdf" />
          </div>
          
          {/* New, improved file status display */}
          <FileStatus files={files} processedFileNames={processedFileNames} />

          <button
            onClick={onProcess}
            disabled={processing || files.length === 0}
            className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-slate-400 flex items-center justify-center transition-colors"
          >
            {processing ? <LoaderCircle className="animate-spin mr-2" size={20} /> : '🚀'}
            {isChatProcessed ? 'Re-process Documents' : 'Process Documents'}
          </button>
        </div>
      </div>
    </aside>
  );
}